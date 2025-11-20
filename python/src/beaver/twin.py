"""Twin: Privacy-preserving dual-value objects for collaborative data science."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

from .live_mixin import LiveMixin
from .remote_data import RemoteData

# Global registry of Twins with private data (for computation resolution)
# Key: (twin_id, owner) -> Twin instance
_TWIN_REGISTRY = {}


@dataclass
class Twin(LiveMixin, RemoteData):
    """
    A dual-value object holding private (real) and public (mock/synthetic) data.

    Extends RemoteData with privacy-preserving dual values and optional live sync.

    Perfect for collaborative data science:
    - Share mock data freely for development
    - Request real data access with explicit approval
    - Privacy-preserving by default
    - Optional live sync for real-time updates

    Examples:
        >>> data = Twin(private="real.csv", public="mock.csv")
        >>> bv.remote_vars["dataset"] = data  # Only public shared!

        >>> # Remote side
        >>> twin = bv.peer("alice").remote_vars["dataset"]
        >>> result = analyze(twin)  # Uses public side
        >>> result.request_private()  # Request real data

        >>> # Enable live sync
        >>> data.enable_live(mutable=False, interval=2.0)
    """

    # Twin-specific IDs (in addition to base id)
    twin_id: str = field(default_factory=lambda: uuid4().hex)
    private_id: str = field(default_factory=lambda: uuid4().hex)
    public_id: str = field(default_factory=lambda: uuid4().hex)

    # Dual values
    private: Optional[Any] = None  # Real data (only if you own it)
    public: Optional[Any] = None  # Mock/synthetic data (always available)

    # Live sync metadata (survives serialization for display)
    live_enabled: bool = False
    live_interval: float = 2.0

    def __post_init__(self):
        """Initialize Twin and validate."""
        # Initialize LiveMixin
        super().__post_init__()

        # Set base id to twin_id for consistency
        if not hasattr(self, "id") or self.id == uuid4().hex:
            object.__setattr__(self, "id", self.twin_id)

        # Validate at least one side
        if self.private is None and self.public is None:
            raise ValueError("Twin must have at least one side (private or public)")

        # Set var_type from data
        if self.var_type == "unknown":
            if self.private is not None:
                object.__setattr__(self, "var_type", f"Twin[{type(self.private).__name__}]")
            elif self.public is not None:
                object.__setattr__(self, "var_type", f"Twin[{type(self.public).__name__}]")

        # Internal state (not serialized)
        self._value_accessed = False
        self._last_value_print = 0.0

        # Live sync subscribers (users who are watching this Twin)
        self._live_subscribers: list[str] = []  # List of usernames
        self._live_context = None  # BeaverContext for sending updates

        # Source tracking for subscribers (where to reload from)
        self._source_path: Optional[str] = None  # Path to reload published data from

        # Register in global registry if this Twin has private data
        if self.private is not None:
            key = (self.twin_id, self.owner)
            _TWIN_REGISTRY[key] = self

    @classmethod
    def public_only(
        cls,
        public: Any,
        owner: str = "unknown",
        live_enabled: bool = False,
        live_interval: float = 2.0,
        **kwargs,
    ):
        """
        Create a Twin with only public data (no private).

        Used when receiving a Twin from remote.
        """
        return cls(
            private=None,
            public=public,
            owner=owner,
            live_enabled=live_enabled,
            live_interval=live_interval,
            **kwargs,
        )

    # ===================================================================
    # RemoteData abstract method implementations
    # ===================================================================

    def has_data(self) -> bool:
        """Check if data is available locally."""
        return self.private is not None or self.public is not None

    def get_value(self) -> Any:
        """Get the preferred value (same as .value property)."""
        return self.value

    # ===================================================================
    # Twin-specific properties
    # ===================================================================

    @property
    def has_private(self) -> bool:
        """Check if private side is available."""
        return self.private is not None

    @property
    def has_public(self) -> bool:
        """Check if public side is available."""
        return self.public is not None

    @property
    def private_value(self):
        """Get private value (raises if not available)."""
        if self.private is None:
            raise ValueError(
                "Private side not available. Call .request_private() to request access from owner."
            )
        return self.private

    @property
    def public_value(self):
        """Get public value (raises if not available)."""
        if self.public is None:
            raise ValueError("Public side not available")
        return self.public

    @property
    def value(self):
        """
        Get the preferred value.

        Prefers private if available (local), else public.
        """
        import time

        # Initialize if not present (deserialized objects)
        if not hasattr(self, "_value_accessed"):
            self._value_accessed = False
            self._last_value_print = 0.0

        # Debounced printing: once per evaluation (1 second debounce)
        current_time = time.time()
        if current_time - self._last_value_print > 1.0:
            if self.has_private:
                print(f"🔒 Using PRIVATE data from Twin '{self.name or self.twin_id[:8]}...'")
            elif self.has_public:
                print(f"🌍 Using PUBLIC data from Twin '{self.name or self.twin_id[:8]}...'")
            self._last_value_print = current_time

        if self.has_private:
            return self.private
        elif self.has_public:
            return self.public
        else:
            raise ValueError("No data available")

    def request_private(self, context=None, name=None):
        """
        Request access to the private side from the owner.

        For data Twins: Creates a request for the owner to share private data.
        For result Twins: Sends the computation request to execute on private data.

        Args:
            context: BeaverContext to use for sending (auto-detected if None)
            name: Override variable name (auto-detected from caller if None)
        """
        import inspect
        from pathlib import Path

        from .computation import ComputationRequest
        from .runtime import _get_var_name_from_caller, pack, write_envelope

        # Check if private is a ComputationRequest (result Twin)
        if isinstance(self.private, ComputationRequest):
            # Auto-detect context if not provided
            if context is None:
                frame = inspect.currentframe()
                if frame and frame.f_back:
                    caller_locals = frame.f_back.f_locals
                    caller_globals = frame.f_back.f_globals
                    # Look for BeaverContext in caller's scope
                    for scope in [caller_locals, caller_globals]:
                        if context:
                            break
                        for _var_name, var_obj in scope.items():
                            if (
                                hasattr(var_obj, "remote_vars")
                                and hasattr(var_obj, "user")
                                and hasattr(var_obj, "inbox_path")
                            ):
                                context = var_obj
                                break

            if not context:
                print("⚠️  No BeaverContext found. Pass context= parameter")
                return

            # Determine the name to use
            # Priority: 1. Explicit name param, 2. Auto-detected variable, 3. Twin's name
            result_name = name
            if result_name is None:
                result_name = _get_var_name_from_caller(self, depth=2)
            if result_name is None:
                result_name = self.name

            print(f"📨 Sending computation request to {self.owner}")
            print(f"   Function: {self.private.func.__name__}")
            print(f"   Result: {result_name}")

            # Pack and send the computation request
            env = pack(
                self.private,
                sender=context.user,
                name=result_name,
            )

            # Send to owner's inbox
            dest_dir = Path(context.outbox).parent / self.owner
            path = write_envelope(env, out_dir=dest_dir)

            print(f"✓ Sent to {path}")
            print(f"💡 Result will auto-update when {self.owner} approves")

            # Store comp_id for auto-update tracking
            self._pending_comp_id = self.private.comp_id

            # Register this Twin for auto-updates
            from .computation import _register_twin_result

            _register_twin_result(self.private.comp_id, self)

        else:
            # Data Twin - request access to private data
            print(f"🔒 Requesting private data access from {self.owner}")
            print(f"   Twin: {self.name or self.twin_id[:8]}")
            print("   💡 (Data access request flow not yet fully implemented)")

    def subscribe_live(self, user: str, context=None):
        """
        Add a user as a live subscriber to this Twin.

        When Live sync is enabled, changes will automatically be sent to subscribers.

        Args:
            user: Username to subscribe
            context: BeaverContext for sending updates (auto-detected if None)
        """
        # Initialize if not present (deserialized objects)
        if not hasattr(self, "_live_subscribers"):
            self._live_subscribers = []
        if not hasattr(self, "_live_context"):
            self._live_context = None

        # Auto-detect context if not provided
        if context is None and self._live_context is None:
            import inspect

            frame = inspect.currentframe()
            while frame and frame.f_back:
                frame = frame.f_back
                for scope in [frame.f_locals, frame.f_globals]:
                    for _var_name, var_obj in scope.items():
                        if (
                            hasattr(var_obj, "remote_vars")
                            and hasattr(var_obj, "user")
                            and hasattr(var_obj, "inbox_path")
                        ):
                            self._live_context = var_obj
                            break
                    if self._live_context:
                        break
                if self._live_context:
                    break

        if context:
            self._live_context = context

        if not self._live_context:
            print("⚠️  Could not find BeaverContext - pass context= parameter")
            return

        if user not in self._live_subscribers:
            self._live_subscribers.append(user)
            print(f"📡 Added '{user}' as live subscriber to Twin '{self.name}'")
            print("💡 Enable live sync to start sending updates: .enable_live()")
        else:
            print(f"⚠️  '{user}' is already subscribed")

    def unsubscribe_live(self, user: str):
        """
        Remove a user from live subscriptions.

        Args:
            user: Username to unsubscribe
        """
        if not hasattr(self, "_live_subscribers"):
            return

        if user in self._live_subscribers:
            self._live_subscribers.remove(user)
            print(f"🔕 Removed '{user}' from live subscribers")
        else:
            print(f"⚠️  '{user}' is not subscribed")

    def watch_live(self, context=None, interval: float = 1.0):
        """
        Watch for live updates from the owner.

        Continuously monitors the inbox for updates to this Twin and
        automatically reloads when changes arrive.

        Args:
            context: BeaverContext to watch (auto-detected if None)
            interval: Check interval in seconds

        Returns:
            Generator that yields the Twin whenever it updates
        """
        import threading
        import time

        # Auto-detect context
        if context is None:
            import inspect

            frame = inspect.currentframe()
            if frame and frame.f_back:
                for scope in [frame.f_back.f_locals, frame.f_back.f_globals]:
                    for _var_name, var_obj in scope.items():
                        if (
                            hasattr(var_obj, "remote_vars")
                            and hasattr(var_obj, "user")
                            and hasattr(var_obj, "inbox_path")
                        ):
                            context = var_obj
                            break
                    if context:
                        break

        if not context:
            print("⚠️  Could not find BeaverContext - pass context= parameter")
            return

        print(f"👁️  Watching for live updates to Twin '{self.name}'...")
        print(f"💡 Updates from '{self.owner}' will auto-reload")

        # Track seen envelope IDs to avoid re-processing
        seen_envelope_ids = set()
        stop_watching = threading.Event()

        try:
            while not stop_watching.is_set():
                # Check inbox for updates
                try:
                    for envelope in context.inbox():
                        # Check if this is an update to our Twin
                        if envelope.name == self.name and envelope.sender == self.owner:
                            # Skip if we've already processed this envelope
                            if envelope.envelope_id in seen_envelope_ids:
                                continue

                            # Mark as seen
                            seen_envelope_ids.add(envelope.envelope_id)

                            # Load the update
                            updated_twin = envelope.load(inject=False)
                            if isinstance(updated_twin, Twin):
                                # Update our Twin's data
                                if updated_twin.has_public:
                                    self.public = updated_twin.public
                                if updated_twin.has_private:
                                    self.private = updated_twin.private

                                # Yield the updated Twin
                                yield self
                except Exception:
                    # Don't crash on inbox errors
                    pass

                # Sleep before next check
                time.sleep(interval)

        except KeyboardInterrupt:
            print(f"\n⚫ Stopped watching Twin '{self.name}'")

    # ===================================================================
    # Delegation methods
    # ===================================================================

    def __len__(self):
        """Delegate len() to preferred value."""
        try:
            return len(self.value)
        except (TypeError, ValueError) as err:
            raise TypeError("Twin has no len()") from err

    def __getitem__(self, key):
        """Delegate indexing to preferred value."""
        return self.value[key]

    # ===================================================================
    # Display methods (unified with RemoteData)
    # ===================================================================

    def __str__(self) -> str:
        """String representation showing both sides with visual state indicators."""
        lines = []
        twin_name = self.name or f"Twin#{self.twin_id[:8]}"

        # Determine state and color scheme
        # Private only (red) = real sensitive data
        # Public only (green) = safe mock data
        # Both (yellow/gold) = be careful which you use
        # Neither/pending (purple) = waiting for data

        if self.has_private and self.has_public:
            # Both sides - yellow/gold warning
            header = f"⚠️  Twin: {twin_name} \033[33m(REAL + MOCK DATA)\033[0m"
            warning = "  ⚠️  \033[33mBe careful: This Twin contains both real and mock data\033[0m"
            lines.append(header)
            lines.append(warning)
        elif self.has_private:
            # Private only - red warning
            header = f"🔒 Twin: {twin_name} \033[31m(REAL DATA - SENSITIVE)\033[0m"
            lines.append(header)
        elif self.has_public:
            # Public only - green safe
            header = f"🌍 Twin: {twin_name} \033[32m(MOCK DATA - SAFE)\033[0m"
            lines.append(header)
        else:
            # Neither - purple pending
            header = f"⏳ Twin: {twin_name} \033[35m(PENDING)\033[0m"
            lines.append(header)

        # Private side
        if self.has_private:
            private_repr = repr(self.private)
            if len(private_repr) > 60:
                private_repr = private_repr[:57] + "..."
            lines.append(f"  \033[31m🔒 Private\033[0m    {private_repr}    ← .value uses this")
        else:
            lines.append("  🔒 Private    (not available) 💡 .request_private()")

        # Public side
        if self.has_public:
            public_repr = repr(self.public)
            if len(public_repr) > 60:
                public_repr = public_repr[:57] + "..."
            active_marker = "    ← .value uses this" if not self.has_private else "    ✓"
            lines.append(f"  \033[32m🌍 Public\033[0m    {public_repr}{active_marker}")
        else:
            lines.append("  🌍 Public    (not set)")

        # Owner (if not local)
        if not self.has_private:
            lines.append(f"  Owner: {self.owner}")

        # Live status (from LiveMixin)
        if hasattr(self, "_display_live_status"):
            lines.extend(self._display_live_status())

        # IDs (compact)
        lines.append(
            f"  IDs: twin={self.twin_id[:8]}... private={self.private_id[:8]}... public={self.public_id[:8]}..."
        )

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Use string representation."""
        return self.__str__()

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter."""
        twin_name = self.name or f"Twin#{self.twin_id[:8]}"

        private_html = ""
        if self.has_private:
            private_repr = repr(self.private)
            if len(private_repr) > 60:
                private_repr = private_repr[:57] + "..."
            private_html = f"<tr><td>🔒 Private</td><td><code>{private_repr}</code></td><td><span style='color: green; font-weight: bold;'>← .value uses this</span></td></tr>"
        else:
            private_html = "<tr><td>🔒 Private</td><td><i>not available</i></td><td><a href='#'>.request_private()</a></td></tr>"

        public_html = ""
        if self.has_public:
            public_repr = repr(self.public)
            if len(public_repr) > 60:
                public_repr = public_repr[:57] + "..."
            active_marker = (
                "<span style='color: green; font-weight: bold;'>← .value uses this</span>"
                if not self.has_private
                else "✓"
            )
            public_html = f"<tr><td>🌍 Public</td><td><code>{public_repr}</code></td><td>{active_marker}</td></tr>"
        else:
            public_html = "<tr><td>🌍 Public</td><td><i>not set</i></td><td></td></tr>"

        owner_html = ""
        if not self.has_private:
            owner_html = f"<tr><td>Owner</td><td colspan='2'><b>{self.owner}</b></td></tr>"

        # Live status
        live_html = ""
        if hasattr(self, "_live_enabled") and self._live_enabled:
            mode = "mutable" if self._live_mutable else "read-only"
            live_html = f"<tr><td>Live</td><td colspan='2'>🟢 Enabled ({mode}, {self._live_interval}s)</td></tr>"
        else:
            live_html = "<tr><td>Live</td><td colspan='2'>⚫ Disabled</td></tr>"

        # Choose border color based on Twin state
        # Red = real/private data (sensitive)
        # Green = public/mock only (safe)
        # Yellow/Orange = both (be careful)
        # Purple = pending/neither (waiting)
        if self.has_private and self.has_public:
            border_color = "#FF9800"  # Orange (warning - both sides)
        elif self.has_private:
            border_color = "#F44336"  # Red (danger - real data)
        elif self.has_public:
            border_color = "#4CAF50"  # Green (safe - mock data)
        else:
            border_color = "#9C27B0"  # Purple (pending)

        return f"""
        <div style='font-family: monospace; border-left: 3px solid {border_color}; padding-left: 10px;'>
            <b>Twin:</b> {twin_name}<br>
            <table style='margin-top: 10px;'>
                {private_html}
                {public_html}
                {owner_html}
                {live_html}
                <tr><td colspan='3' style='padding-top: 10px; font-size: 10px; color: #666;'>
                    IDs: twin={self.twin_id[:8]}... private={self.private_id[:8]}... public={self.public_id[:8]}...
                </td></tr>
            </table>
        </div>
        """
