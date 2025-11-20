"""Remote computation tracking and execution."""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from uuid import uuid4


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Global registry of active computation pointers
_COMPUTATION_POINTERS = {}


def _register_computation_pointer(pointer):
    """Register a computation pointer for auto-updates."""
    _COMPUTATION_POINTERS[pointer.comp_id] = pointer


def _update_computation_pointer(comp_id: str, result_value: Any):
    """Update a computation pointer with its result."""
    if comp_id in _COMPUTATION_POINTERS:
        pointer = _COMPUTATION_POINTERS[comp_id]
        pointer._result_value = result_value
        pointer.status = "complete"
        pointer.completed_at = _iso_now()
        return True
    return False


# Global registry of Twin results awaiting updates
_TWIN_RESULTS = {}


def _register_twin_result(comp_id: str, twin):
    """Register a Twin result for auto-updates when computation completes."""
    _TWIN_RESULTS[comp_id] = twin


def _update_twin_result(comp_id: str, result_value: Any):
    """Update a Twin's private side with the computation result."""
    if comp_id in _TWIN_RESULTS:
        twin = _TWIN_RESULTS[comp_id]
        # Update the private side with the actual result
        object.__setattr__(twin, "private", result_value)
        print(f"✨ Twin result auto-updated: {twin.name or twin.twin_id[:8]}...")
        print("   .value now uses private (approved result)")
        # Remove from registry after updating
        del _TWIN_RESULTS[comp_id]
        return True
    return False


@dataclass
class ComputationResult:
    """
    Result of a computation execution with approval workflow.

    Allows reviewing, modifying, and sending results back to requester.
    """

    result: Any
    stdout: str
    stderr: str
    error: Optional[str]
    var_name: str
    var_id: str
    comp_id: str
    sender: str
    context: Any = None

    @property
    def data(self):
        """Get the result data."""
        return self.result

    @data.setter
    def data(self, value):
        """Modify the result data before approval."""
        self.result = value

    def approve(self):
        """
        Approve and send the private/real result back to the requester.

        If result is a Twin, sends the private side only.
        """
        from .twin import Twin

        if isinstance(self.result, Twin):
            if self.result.has_private:
                print(f"✅ Approving private/real result for: {self.var_name}")
                return self._send_result(self.result.private)
            else:
                raise ValueError("Twin has no private data to approve")
        else:
            return self._send_result(self.result)

    def approve_mock(self):
        """
        Approve and send only the mock/public result back.

        Useful for iterative development - lets requester continue working
        with mock data while you review the private computation.
        """
        from .twin import Twin

        if isinstance(self.result, Twin):
            if self.result.has_public:
                print(f"🧪 Approving mock/public result for: {self.var_name}")
                print("   💡 Requester can continue development with mock data")
                return self._send_result(self.result.public)
            else:
                raise ValueError("Twin has no public data to approve")
        else:
            raise ValueError("Result is not a Twin - use .approve() instead")

    def approve_with(self, value):
        """Approve with a substituted value."""
        return self._send_result(value)

    def reject(self, message="Request rejected"):
        """
        Reject the computation request with an explanation.

        Sends a rejection message back to the requester.
        """
        if not self.context:
            raise ValueError("No context available for sending rejection")

        print(f"❌ Rejecting computation: {self.var_name}")
        print(f"   Reason: {message}")
        print(f"   Notifying: {self.sender}")

        # Create rejection message
        rejection = {
            "_beaver_rejection": True,
            "comp_id": self.comp_id,
            "result_name": self.var_name,
            "message": message,
            "rejected_at": _iso_now(),
        }

        # Send rejection back
        result = self.context.send(
            rejection,
            name=f"rejection_{self.var_name}",
            user=self.sender,
            reply_to=self.comp_id,
        )

        print(f"✓ Rejection sent to {self.sender}'s inbox")
        return result

    def _send_result(self, value):
        """Send the result back to the requester."""
        if not self.context:
            raise ValueError("No context available for sending result")

        print(f"✅ Approving result for: {self.var_name}")
        print(f"   Sending to: {self.sender}")

        # Send the result back as a reply
        result = self.context.send(
            value,
            name=self.var_name,
            user=self.sender,
            reply_to=self.comp_id,
        )

        print(f"✓ Result sent to {self.sender}'s inbox")
        print(f"💡 They can access it with bv.inbox()['{self.var_name}'].load()")

        return result

    def __repr__(self) -> str:
        """String representation - shows Twin directly with context."""
        from .twin import Twin

        # If result is a Twin, show it directly with minimal wrapper
        if isinstance(self.result, Twin):
            lines = []
            lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            lines.append(f"ComputationResult: {self.var_name}")
            lines.append(f"  Request from: {self.sender}")
            lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            lines.append("")

            # Show the Twin directly (use its __str__)
            twin_str = str(self.result)
            lines.append(twin_str)

            # Add execution info if available
            if self.error:
                lines.append(f"\n  ❌ Execution Error: {self.error}")

            if self.stdout and self.stdout.strip():
                lines.append("\n  📤 Output captured during execution:")
                for line in self.stdout.strip().split("\n")[:5]:  # Show first 5 lines
                    lines.append(f"     {line}")
                if len(self.stdout.strip().split("\n")) > 5:
                    more_lines = len(self.stdout.strip().split("\n")) - 5
                    lines.append(f"     ... ({more_lines} more lines)")

            # Show actions based on Twin state
            lines.append("")
            lines.append("  💡 Actions:")
            if self.result.has_private and self.result.has_public:
                lines.append("     .approve()           - Send private/real result back")
                lines.append("     .approve_mock()      - Send mock result (continue iteration)")
            elif self.result.has_private:
                lines.append("     .approve()           - Send private/real result back")
            elif self.result.has_public:
                lines.append("     .approve_mock()      - Send mock result back")

            lines.append("     .reject(message)     - Reject with explanation")
            lines.append("     .data                - Access Twin to inspect/modify")

            return "\n".join(lines)

        # Non-Twin result (legacy/error case)
        else:
            lines = [
                f"ComputationResult: {self.var_name}",
                f"  From request by: {self.sender}",
                f"  Result ID: {self.var_id[:12]}...",
                f"  Comp ID: {self.comp_id[:12]}...",
            ]

            if self.error:
                lines.append(f"  ❌ Error: {self.error}")
            else:
                result_type = type(self.result).__name__
                result_repr = repr(self.result)
                if len(result_repr) > 60:
                    result_repr = result_repr[:57] + "..."
                lines.append(f"  ✓ Result ({result_type}): {result_repr}")

            if self.stdout:
                lines.append(f"  📤 Stdout: {len(self.stdout)} chars")
            if self.stderr:
                lines.append(f"  ⚠️  Stderr: {len(self.stderr)} chars")

            lines.append("")
            lines.append("  💡 Actions:")
            lines.append("     .approve()           - Send result back")
            lines.append("     .reject(message)     - Reject with explanation")

            return "\n".join(lines)


@dataclass
class ComputationRequest:
    """
    A request to execute a computation remotely.

    This is the serializable object sent to the remote user's inbox.
    """

    comp_id: str
    result_id: str
    func: Any  # The callable function
    args: tuple
    kwargs: dict
    sender: str
    result_name: str
    created_at: str = field(default_factory=_iso_now)

    def _auto_detect_context(self):
        """Auto-detect BeaverContext from caller's scope."""
        import inspect

        # Try to find the expected user from remote var args
        expected_user = None
        for arg in self.args:
            if isinstance(arg, dict) and arg.get("_beaver_remote_var"):
                expected_user = arg["owner"]
                break

        # Walk up the frame stack to find a matching BeaverContext
        frame = inspect.currentframe()
        candidates = []
        while frame and frame.f_back:
            frame = frame.f_back
            # Collect all BeaverContext objects
            for scope in [frame.f_locals, frame.f_globals]:
                for _var_name, var_obj in scope.items():
                    if (
                        hasattr(var_obj, "remote_vars")
                        and hasattr(var_obj, "user")
                        and hasattr(var_obj, "inbox_path")
                    ):
                        candidates.append(var_obj)

        # Prefer context that matches expected user, otherwise first found
        if expected_user:
            for candidate in candidates:
                if candidate.user == expected_user:
                    return candidate

        if candidates:
            return candidates[0]

        return None

    def execute(self, context=None) -> dict:
        """
        Execute this computation request.

        Args:
            context: BeaverContext for the executing user (auto-detected if None)

        Returns:
            Result dictionary with result, stdout, stderr, error
        """
        # Auto-detect context from caller's scope if not provided
        if context is None:
            import inspect

            frame = inspect.currentframe()

            # Try to find the expected user from remote var args
            expected_user = None
            for arg in self.args:
                if isinstance(arg, dict) and arg.get("_beaver_remote_var"):
                    expected_user = arg["owner"]
                    break

            # Walk up the frame stack to find a matching BeaverContext
            current = frame
            candidates = []
            while current and current.f_back:
                current = current.f_back
                # Collect all BeaverContext objects
                for scope in [current.f_locals, current.f_globals]:
                    for _var_name, var_obj in scope.items():
                        if (
                            hasattr(var_obj, "remote_vars")
                            and hasattr(var_obj, "user")
                            and hasattr(var_obj, "inbox_path")
                        ):
                            candidates.append(var_obj)

            # Prefer context that matches expected user, otherwise first found
            if expected_user:
                for candidate in candidates:
                    if candidate.user == expected_user:
                        context = candidate
                        break

            if context is None and candidates:
                context = candidates[0]

        print(f"⚙️  Executing: {self.result_name} = {self.func.__name__}(...)")
        print(f"   From: {self.sender}")
        print(f"   Computation ID: {self.comp_id[:12]}...")
        if context:
            print(f"   Context: {context.user}")

        result_data = execute_remote_computation(
            func=self.func,
            args=self.args,
            kwargs=self.kwargs,
            result_var_name=self.result_name,
            result_var_id=self.result_id,
            capture_output=True,
            context=context,
        )

        print("✓ Execution complete")
        if result_data["error"]:
            print(f"❌ Error: {result_data['error']}")
        else:
            print(f"✓ Result: {type(result_data['result']).__name__}")

        # Return a ComputationResult object instead of raw dict
        return ComputationResult(
            result=result_data["result"],
            stdout=result_data["stdout"],
            stderr=result_data["stderr"],
            error=result_data["error"],
            var_name=self.result_name,
            var_id=self.result_id,
            comp_id=self.comp_id,
            sender=self.sender,
            context=context,
        )

    def run(self, context=None):
        """
        Execute the computation on real/private data only.

        Returns ComputationResult where .data is a Twin with only private side.
        If execution fails, returns ComputationResult with error (no Twin).
        """
        # Auto-detect context if needed
        if context is None:
            context = self._auto_detect_context()

        comp_result = self.execute(context=context)

        # Only wrap in Twin if execution succeeded
        if comp_result.error is None and comp_result.result is not None:
            from .twin import Twin

            result_twin = Twin(
                private=comp_result.result,
                public=None,
                owner=context.user if context else "unknown",
                name=self.result_name,
            )
            comp_result.result = result_twin

        return comp_result

    def run_mock(self, context=None):
        """
        Execute the computation on mock/public data for safe preview.

        Returns ComputationResult where .data is a Twin with only public side.
        """
        from .twin import Twin

        # Auto-detect context if needed
        if context is None:
            context = self._auto_detect_context()

        # Replace Twin arguments with their public sides
        mock_args = []
        for arg in self.args:
            if isinstance(arg, Twin):
                if arg.has_public:
                    mock_args.append(arg.public)
                else:
                    raise ValueError("Twin argument has no public data for mock testing")
            else:
                mock_args.append(arg)

        mock_kwargs = {}
        for k, v in self.kwargs.items():
            if isinstance(v, Twin):
                if v.has_public:
                    mock_kwargs[k] = v.public
                else:
                    raise ValueError(f"Twin kwarg '{k}' has no public data for mock testing")
            else:
                mock_kwargs[k] = v

        # Execute on mock data
        print("🧪 Testing on mock/public data...")
        mock_result = self.func(*mock_args, **mock_kwargs)
        print(f"✓ Mock result: {type(mock_result).__name__}")

        # Wrap in Twin with only public side
        result_twin = Twin(
            private=None,
            public=mock_result,
            owner=context.user if context else "unknown",
            name=self.result_name,
        )

        # Return ComputationResult
        return ComputationResult(
            result=result_twin,
            stdout="",
            stderr="",
            error=None,
            var_name=self.result_name,
            var_id=self.result_id,
            comp_id=self.comp_id,
            sender=self.sender,
            context=context,
        )

    def run_both(self, context=None):
        """
        Execute on both mock and real data.

        Runs mock first, then real. Returns ComputationResult where .data is
        a Twin with both sides for comparison.
        """
        from .twin import Twin

        # Auto-detect context if needed
        if context is None:
            context = self._auto_detect_context()

        # Run mock first
        print("🧪 Step 1/2: Testing on mock/public data...")
        mock_comp_result = self.run_mock(context=context)

        # Then run on real data
        print("🔒 Step 2/2: Executing on real/private data...")
        comp_result = self.execute(context=context)

        # Extract the mock data from the Twin returned by run_mock()
        mock_data = (
            mock_comp_result.result.public
            if isinstance(mock_comp_result.result, Twin)
            else mock_comp_result.result
        )

        # Create Twin with both sides
        result_twin = Twin(
            public=mock_data,
            private=comp_result.result,
            owner=context.user if context else "unknown",
            name=self.result_name,
        )

        # Update the ComputationResult's data field to be the Twin
        comp_result.result = result_twin
        return comp_result

    def __call__(self, context=None):
        """Allow calling the request directly."""
        return self.execute(context=context)

    def __repr__(self) -> str:
        """String representation - beautiful Action display."""
        lines = []

        # Header
        lines.append("━" * 70)
        lines.append(f"⚡ Action: {self.result_name}")
        lines.append(f"   Request from: {self.sender}")
        lines.append("━" * 70)

        # Function info
        lines.append("")
        lines.append(f"📋 Function: \033[36m{self.func.__name__}\033[0m")

        # Bound Data section - show what data is bound to this action
        has_data = False
        data_lines = []

        for _i, arg in enumerate(self.args):
            from .twin import Twin

            if isinstance(arg, Twin) or (isinstance(arg, dict) and arg.get("_beaver_remote_var")):
                has_data = True

                if isinstance(arg, Twin):
                    # Determine privacy level and color
                    if arg.has_private and arg.has_public:
                        privacy_badge = "\033[33m⚠️  REAL+MOCK\033[0m"
                        indent_color = "\033[33m"  # Yellow
                    elif arg.has_private:
                        privacy_badge = "\033[31m🔒 PRIVATE\033[0m"
                        indent_color = "\033[31m"  # Red
                    elif arg.has_public:
                        privacy_badge = "\033[32m🌍 PUBLIC\033[0m"
                        indent_color = "\033[32m"  # Green
                    else:
                        privacy_badge = "\033[35m⏳ PENDING\033[0m"
                        indent_color = "\033[35m"  # Purple

                    # Extract underlying type from Twin[type]
                    underlying_type = (
                        arg.var_type.replace("Twin[", "").replace("]", "")
                        if arg.var_type.startswith("Twin[")
                        else arg.var_type
                    )

                    data_lines.append(f"  {indent_color}│\033[0m {privacy_badge}")
                    data_lines.append(
                        f"  {indent_color}│\033[0m   Parameter: \033[36m{arg.name or 'unnamed'}\033[0m"
                    )
                    data_lines.append(f"  {indent_color}│\033[0m   Type: {underlying_type}")
                    data_lines.append(f"  {indent_color}│\033[0m   Owner: {arg.owner}")

                    if arg.has_public:
                        data_lines.append(
                            f"  {indent_color}│\033[0m   📊 Mock data available for testing"
                        )
                    if arg.has_private:
                        data_lines.append(
                            f"  {indent_color}│\033[0m   🔐 Real data available (you own this)"
                        )

                elif isinstance(arg, dict) and arg.get("_beaver_remote_var"):
                    # RemoteVar reference
                    var_type = arg.get("var_type", "unknown")
                    is_twin = var_type.startswith("Twin[")

                    if is_twin:
                        underlying_type = var_type.replace("Twin[", "").replace("]", "")
                        privacy_badge = "\033[35m🔗 TWIN REF\033[0m"
                        indent_color = "\033[35m"
                    else:
                        underlying_type = var_type
                        privacy_badge = "\033[36m🔗 DATA REF\033[0m"
                        indent_color = "\033[36m"

                    data_lines.append(f"  {indent_color}│\033[0m {privacy_badge}")
                    data_lines.append(
                        f"  {indent_color}│\033[0m   Parameter: \033[36m{arg['name']}\033[0m"
                    )
                    data_lines.append(f"  {indent_color}│\033[0m   Type: {underlying_type}")
                    data_lines.append(f"  {indent_color}│\033[0m   Owner: {arg['owner']}")

        # Check kwargs too
        for _k, v in self.kwargs.items():
            from .twin import Twin

            if isinstance(v, Twin) or (isinstance(v, dict) and v.get("_beaver_remote_var")):
                has_data = True
                # Similar logic for kwargs...

        if has_data:
            lines.append("")
            lines.append("📦 Bound Data:")
            lines.extend(data_lines)

        # Static parameters
        static_args = []
        for i, arg in enumerate(self.args):
            from .twin import Twin

            if not isinstance(arg, Twin) and not (
                isinstance(arg, dict) and arg.get("_beaver_remote_var")
            ):
                arg_type = type(arg).__name__
                arg_repr = repr(arg)
                if len(arg_repr) > 50:
                    arg_repr = arg_repr[:47] + "..."
                static_args.append(f"  • arg[{i}]: {arg_type} = {arg_repr}")

        static_kwargs = []
        for k, v in self.kwargs.items():
            from .twin import Twin

            if not isinstance(v, Twin) and not (
                isinstance(v, dict) and v.get("_beaver_remote_var")
            ):
                v_type = type(v).__name__
                v_repr = repr(v)
                if len(v_repr) > 50:
                    v_repr = v_repr[:47] + "..."
                static_kwargs.append(f"  • {k}: {v_type} = {v_repr}")

        if static_args or static_kwargs:
            lines.append("")
            lines.append("⚙️  Static Parameters:")
            lines.extend(static_args)
            lines.extend(static_kwargs)

        # Actions
        lines.append("")
        lines.append("💡 Next Steps:")
        lines.append("   .run_mock()   → Test on mock/public data (safe preview)")
        lines.append("   .run()        → Execute on real/private data")
        lines.append("   .run_both()   → Run on both mock & real for comparison")
        lines.append("   .reject(msg)  → Decline this request")

        lines.append("")
        lines.append(f"🆔 IDs: comp={self.comp_id[:12]}... result={self.result_id[:12]}...")
        lines.append("━" * 70)

        return "\n".join(lines)


@dataclass
class RemoteComputationPointer:
    """
    Pointer to a computation that will execute remotely.

    Represents: result = func(*args, **kwargs)
    But execution is deferred until sent to remote.

    Auto-updates when result arrives!
    """

    comp_id: str = field(default_factory=lambda: uuid4().hex)
    result_id: str = field(default_factory=lambda: uuid4().hex)
    func: Optional[Callable] = None
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    destination: str = ""  # Which user will execute this
    result_name: str = "result"  # Name for result variable
    status: str = "pending"  # pending, staged, sent, running, complete, error
    created_at: str = field(default_factory=_iso_now)
    sent_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    context: Optional[Any] = None  # BeaverContext reference
    _result_value: Optional[Any] = None  # Holds result when it arrives

    def __post_init__(self):
        """Register this pointer for auto-updates."""
        _register_computation_pointer(self)

    def stage(self):
        """Stage this computation for sending (git-style)."""
        if self.status == "pending":
            self.status = "staged"
            if self.context:
                self.context._add_staged(self)
            print(f"📦 Staged: {self.result_name} = {self._func_name()}(...)")
            print(f"   Destination: {self.destination}")
            print("   Use bv.send_staged() to send all")
        return self

    def send(self, *, wait: bool = False):
        """Send this computation immediately (skip staging)."""
        if not self.context:
            raise ValueError("No context available for sending")

        print(f"📤 Sending: {self.result_name} = {self._func_name()}(...)")
        print(f"   → {self.destination}")

        # Convert RemoteVarPointers to serializable references
        from .remote_vars import RemoteVarPointer

        def convert_arg(arg):
            """Convert RemoteVarPointer to var reference."""
            if isinstance(arg, RemoteVarPointer):
                # Return a dict with owner and var_id for remote resolution
                return {
                    "_beaver_remote_var": True,
                    "owner": arg.remote_var.owner,
                    "var_id": arg.remote_var.var_id,
                    "name": arg.remote_var.name,
                }
            return arg

        serializable_args = tuple(convert_arg(arg) for arg in self.args)
        serializable_kwargs = {k: convert_arg(v) for k, v in self.kwargs.items()}

        # Create a computation request object
        comp_request = ComputationRequest(
            comp_id=self.comp_id,
            result_id=self.result_id,
            func=self.func,
            args=serializable_args,
            kwargs=serializable_kwargs,
            sender=self.context.user,
            result_name=self.result_name,
        )

        # Pack and send the computation request
        from pathlib import Path

        from .runtime import pack, write_envelope

        envelope = pack(
            comp_request,
            sender=self.context.user,
            name=f"computation_{self.result_name}",
            strict=self.context.strict,
            policy=self.context.policy,
        )

        # Write to destination's inbox
        dest_dir = Path("shared") / self.destination
        path = write_envelope(envelope, out_dir=dest_dir)

        self.status = "sent"
        self.sent_at = _iso_now()
        print(f"✓ Sent to {path}")
        print(f"⏳ Waiting for {self.destination} to execute")

        if wait:
            print("⚠️  .wait() not yet implemented")

        return self

    def wait(self, timeout: float = 30.0):
        """Wait for computation to complete."""
        # TODO: Poll for result
        print(f"⏳ Waiting for result (timeout={timeout}s)...")
        print("⚠️  Not yet implemented")
        return None

    def refresh(self):
        """Check if computation completed and update status."""
        # TODO: Check remote vars for result
        if self.status in ("sent", "running"):
            print(f"🔄 Checking status of {self.result_name}...")
            print("⚠️  Not yet implemented")
        return self

    def _func_name(self) -> str:
        """Get function name for display."""
        if self.func:
            return getattr(self.func, "__name__", str(self.func))
        return "unknown"

    @property
    def value(self):
        """Get the result value if available."""
        if self._result_value is not None:
            return self._result_value
        elif self.status == "complete":
            return self._result_value  # Could be None
        else:
            raise ValueError(
                f"Result not available yet (status: {self.status}). "
                f"Call .refresh() to check for updates."
            )

    def has_result(self) -> bool:
        """Check if result is available."""
        return self._result_value is not None or self.status == "complete"

    def __repr__(self) -> str:
        """String representation."""
        status_emoji = {
            "pending": "⏸️",
            "staged": "📦",
            "sent": "📤",
            "running": "⚙️",
            "complete": "✅",
            "error": "❌",
        }
        emoji = status_emoji.get(self.status, "❓")

        lines = [
            f"RemoteComputationPointer: {self.result_name}",
            f"  {emoji} Status: {self.status}",
            f"  🎯 Destination: {self.destination}",
            f"  🔧 Function: {self._func_name()}",
            f"  🆔 Result ID: {self.result_id[:12]}...",
        ]

        if self.status == "pending":
            lines.append("  💡 Call .stage() or .send()")
        elif self.status == "staged":
            lines.append("  💡 Call bv.send_staged() to send")
        elif self.status == "sent":
            lines.append(f"  💡 Sent at {self.sent_at[:19]}")
            lines.append("  💡 Waiting for result... (auto-updates)")
        elif self.status == "complete":
            lines.append(f"  ✓ Completed at {self.completed_at[:19]}")
            if self._result_value is not None:
                result_repr = repr(self._result_value)
                if len(result_repr) > 60:
                    result_repr = result_repr[:57] + "..."
                lines.append(f"  ✓ Result: {result_repr}")
                lines.append("  💡 Access with .value")

        if self.error:
            lines.append(f"  ❌ Error: {self.error}")

        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter."""
        status_color = {
            "pending": "#FFA500",
            "staged": "#2196F3",
            "sent": "#9C27B0",
            "running": "#FF9800",
            "complete": "#4CAF50",
            "error": "#F44336",
        }
        color = status_color.get(self.status, "#757575")

        return (
            f"<div style='border-left: 3px solid {color}; padding-left: 10px; font-family: monospace;'>"
            f"<b>RemoteComputationPointer:</b> {self.result_name}<br>"
            f"<b>Status:</b> <span style='color: {color};'>{self.status}</span><br>"
            f"<b>Destination:</b> {self.destination}<br>"
            f"<b>Function:</b> {self._func_name()}<br>"
            f"<b>Result ID:</b> <code>{self.result_id[:12]}...</code>"
            f"</div>"
        )


def execute_remote_computation(
    func: Callable,
    args: tuple,
    kwargs: dict,
    result_var_name: str,
    result_var_id: str,
    capture_output: bool = True,
    context=None,
) -> dict:
    """
    Execute a remote computation with output capture.

    Args:
        func: The function to execute
        args: Positional arguments (may include RemoteVar references to resolve)
        kwargs: Keyword arguments
        result_var_name: Name for the result variable
        result_var_id: Pre-assigned ID for the result
        capture_output: Whether to capture stdout/stderr
        context: BeaverContext for resolving remote vars

    Returns:
        Dict with result, output, error info
    """

    def resolve_arg(arg):
        """Resolve remote var references and Twins to actual values."""
        # Check if this is a remote var reference dict
        if isinstance(arg, dict) and arg.get("_beaver_remote_var"):
            if context:
                # Look up the var by ID (not name, since param names != var names)
                var_id = arg["var_id"]
                for var in context.remote_vars.vars.values():
                    if var.var_id == var_id:
                        if var._stored_value is not None:
                            return var._stored_value
                        else:
                            raise ValueError(
                                f"Remote var '{var.name}' (ID: {var_id[:12]}...) "
                                f"has no stored value"
                            )
                raise ValueError(f"Remote var ID {var_id[:12]}... not found in registry")
            else:
                raise ValueError("Cannot resolve remote var reference without context")

        # Check if this is a Twin - look it up in global Twin registry
        from .twin import _TWIN_REGISTRY, Twin

        if isinstance(arg, Twin):
            if context:
                # Try to find the owner's version of this Twin by (twin_id, owner)
                twin_id = arg.twin_id
                owner = arg.owner

                # Check global Twin registry for the executing user's version
                key = (twin_id, context.user)
                if key in _TWIN_REGISTRY:
                    registered_twin = _TWIN_REGISTRY[key]
                    if registered_twin.has_private:
                        return registered_twin

                # Also check with arg's owner (in case it differs from context.user)
                if owner != context.user:
                    key = (twin_id, owner)
                    if key in _TWIN_REGISTRY:
                        registered_twin = _TWIN_REGISTRY[key]
                        if registered_twin.has_private:
                            return registered_twin

            # Use the received Twin if no local version with private data found
            return arg

        return arg

    # Resolve any remote var references in args/kwargs
    resolved_args = [resolve_arg(arg) for arg in args]

    result_data = {
        "result": None,
        "stdout": "",
        "stderr": "",
        "error": None,
        "var_name": result_var_name,
        "var_id": result_var_id,
    }

    try:
        if capture_output:
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                result = func(*resolved_args, **kwargs)

            result_data["stdout"] = stdout_capture.getvalue()
            result_data["stderr"] = stderr_capture.getvalue()
        else:
            result = func(*resolved_args, **kwargs)

        result_data["result"] = result

    except Exception as e:
        result_data["error"] = str(e)
        import traceback

        result_data["stderr"] = traceback.format_exc()

    return result_data


class StagingArea:
    """
    Git-style staging area for remote computations.
    """

    def __init__(self, context):
        self.context = context
        self.staged: list[RemoteComputationPointer] = []

    def add(self, computation: RemoteComputationPointer):
        """Add a computation to staging."""
        if computation not in self.staged:
            self.staged.append(computation)

    def remove(self, computation: RemoteComputationPointer):
        """Remove from staging."""
        if computation in self.staged:
            self.staged.remove()

    def clear(self):
        """Clear all staged computations."""
        self.staged.clear()

    def send_all(self):
        """Send all staged computations."""
        if not self.staged:
            print("📭 No staged computations to send")
            return

        print(f"📦 Sending {len(self.staged)} staged computation(s)...")
        for comp in self.staged:
            # Actually send each computation
            comp.send()

        self.staged.clear()
        print("✅ All staged computations sent")

    def __repr__(self) -> str:
        if not self.staged:
            return "StagingArea: empty"

        lines = ["StagingArea:"]
        for comp in self.staged:
            lines.append(f"  📦 {comp.result_name} = {comp._func_name()}(...) → {comp.destination}")
        lines.append(f"\n💡 Call bv.send_staged() to send {len(self.staged)} item(s)")

        return "\n".join(lines)

    def _repr_html_(self) -> str:
        if not self.staged:
            return "<b>StagingArea:</b> empty"

        rows = []
        for comp in self.staged:
            rows.append(
                f"<tr>"
                f"<td>{comp.result_name}</td>"
                f"<td>{comp._func_name()}(...)</td>"
                f"<td>{comp.destination}</td>"
                f"<td><code>{comp.result_id[:8]}...</code></td>"
                f"</tr>"
            )

        return (
            f"<b>StagingArea</b> ({len(self.staged)} staged)<br>"
            f"<table>"
            f"<thead><tr><th>Result</th><th>Function</th><th>Destination</th><th>ID</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            f"</table>"
            f"<p>💡 Call <code>bv.send_staged()</code> to send</p>"
        )
