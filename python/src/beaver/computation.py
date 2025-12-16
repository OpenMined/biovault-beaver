"""Remote computation tracking and execution."""

from __future__ import annotations

import ast
import builtins
import inspect
import io
from contextlib import redirect_stderr, redirect_stdout, suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from uuid import uuid4


def _detect_function_imports(func: Callable) -> list[str]:
    """
    Detect import statements inside a function.

    Returns a list of top-level module names that the function imports.
    """
    import textwrap

    try:
        source = inspect.getsource(func)
        # Dedent to handle nested functions (preserves relative indentation)
        source = textwrap.dedent(source)
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError):
        return []

    # Find the function definition
    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func.__name__:
            func_def = node
            break

    if func_def is None:
        return []

    # Collect imports - get the top-level module name
    imports = []
    for node in ast.walk(func_def):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # Get top-level module (e.g., "matplotlib" from "matplotlib.pyplot")
                top_module = alias.name.split(".")[0]
                if top_module not in imports:
                    imports.append(top_module)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_module = node.module.split(".")[0]
            if top_module not in imports:
                imports.append(top_module)

    return imports


def _detect_function_imports_with_versions(func: Callable) -> dict[str, str | None]:
    """
    Detect import statements inside a function and get their installed versions.

    Returns a dict mapping module names to their versions (or None if not installed).
    """
    import importlib.metadata

    imports = _detect_function_imports(func)
    result = {}

    # Map common module names to pip package names for version lookup
    package_map = {
        "sklearn": "scikit-learn",
        "cv2": "opencv-python",
        "PIL": "pillow",
        "skimage": "scikit-image",
    }

    for module_name in imports:
        pkg_name = package_map.get(module_name, module_name)
        try:
            version = importlib.metadata.version(pkg_name)
            result[module_name] = version
        except importlib.metadata.PackageNotFoundError:
            # Try the module name directly
            try:
                version = importlib.metadata.version(module_name)
                result[module_name] = version
            except importlib.metadata.PackageNotFoundError:
                result[module_name] = None

    return result


def _check_missing_imports(func: Callable) -> list[str]:
    """
    Check which imports in a function are not installed.

    Returns list of missing package names.
    """
    imports = _detect_function_imports(func)
    missing = []

    for module_name in imports:
        # Map common module names to pip package names
        package_map = {
            "sklearn": "scikit-learn",
            "cv2": "opencv-python",
            "PIL": "pillow",
            "skimage": "scikit-image",
        }
        try:
            __import__(module_name)
        except ImportError:
            pkg_name = package_map.get(module_name, module_name)
            if pkg_name not in missing:
                missing.append(pkg_name)

    return missing


def _check_missing_imports_with_versions(
    func: Callable, required_versions: dict[str, str] | None = None
) -> list[tuple[str, str | None]]:
    """
    Check which imports in a function are not installed.

    Args:
        func: The function to check
        required_versions: Optional dict of module_name -> version from sender

    Returns list of (package_name, required_version) tuples for missing packages.
    """
    imports = _detect_function_imports(func)
    missing = []

    package_map = {
        "sklearn": "scikit-learn",
        "cv2": "opencv-python",
        "PIL": "pillow",
        "skimage": "scikit-image",
    }

    for module_name in imports:
        try:
            __import__(module_name)
        except ImportError:
            pkg_name = package_map.get(module_name, module_name)
            version = required_versions.get(module_name) if required_versions else None
            if not any(p[0] == pkg_name for p in missing):
                missing.append((pkg_name, version))

    return missing


def _is_uv_venv() -> bool:
    """Check if we're in a uv-managed virtual environment."""
    import os
    import shutil
    import sys
    from pathlib import Path

    if hasattr(sys, "prefix"):
        uv_marker = Path(sys.prefix) / "uv.lock"
        if uv_marker.exists():
            return True
        pyvenv_cfg = Path(sys.prefix) / "pyvenv.cfg"
        if pyvenv_cfg.exists():
            with suppress(Exception):
                content = pyvenv_cfg.read_text()
                if "uv" in content.lower():
                    return True
    if os.environ.get("UV_VIRTUAL_ENV"):
        return True
    return bool(os.environ.get("VIRTUAL_ENV") and shutil.which("uv"))


def _prompt_install_function_deps(
    missing: list[str] | list[tuple[str, str | None]], func_name: str
) -> bool:
    """
    Prompt user to install missing dependencies for a function.

    Args:
        missing: Either a list of package names, or list of (package_name, version) tuples
        func_name: Name of the function requiring these deps

    Returns True if installation succeeded, False otherwise.
    """
    import shutil
    import subprocess
    import sys

    use_uv = _is_uv_venv() and shutil.which("uv")
    pip_cmd = "uv pip install" if use_uv else "pip install"

    # Normalize to list of (pkg, version) tuples
    if missing and isinstance(missing[0], str):
        missing_with_versions: list[tuple[str, str | None]] = [(p, None) for p in missing]
    else:
        missing_with_versions = missing  # type: ignore

    # Build install specs with versions where available
    install_specs = []
    display_names = []
    for pkg, version in missing_with_versions:
        if version:
            install_specs.append(f"{pkg}=={version}")
            display_names.append(f"{pkg}=={version}")
        else:
            install_specs.append(pkg)
            display_names.append(pkg)

    deps_str = " ".join(install_specs)

    print(f"\n⚠️  Function '{func_name}' requires packages not installed:")
    for name in display_names:
        print(f"   • {name}")
    print()

    try:
        response = input(f"Install now with '{pip_cmd} {deps_str}'? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        print("💡 To install manually, run:")
        print(f"   {pip_cmd} {deps_str}")
        return False

    if response in ("y", "yes"):
        print(f"\n📦 Installing {deps_str}...")
        try:
            if use_uv:
                cmd = ["uv", "pip", "install"] + install_specs
            else:
                cmd = [sys.executable, "-m", "pip", "install"] + install_specs
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print(f"✅ Successfully installed: {deps_str}")
                # Re-import to make packages available
                for pkg, _ in missing_with_versions:
                    with suppress(Exception):
                        __import__(pkg)
                return True
            else:
                print("❌ Installation failed:")
                print(result.stderr)
                print("\n💡 To install manually, run:")
                print(f"   {pip_cmd} {deps_str}")
                return False
        except Exception as e:
            print(f"❌ Installation failed: {e}")
            print("\n💡 To install manually, run:")
            print(f"   {pip_cmd} {deps_str}")
            return False
    else:
        print("\n💡 To install manually, run:")
        print(f"   {pip_cmd} {deps_str}")
        return False


def _detect_global_access(func: Callable) -> list[str]:
    """
    Detect potential global/non-local variable access in a function.

    Returns a list of variable names that appear to be accessed from outer scope.
    """
    try:
        source = inspect.getsource(func)
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError):
        return []

    # Find the function definition
    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func.__name__:
            func_def = node
            break

    if func_def is None:
        return []

    # Collect parameter names
    params = set()
    for arg in func_def.args.args:
        params.add(arg.arg)
    for arg in func_def.args.posonlyargs:
        params.add(arg.arg)
    for arg in func_def.args.kwonlyargs:
        params.add(arg.arg)
    if func_def.args.vararg:
        params.add(func_def.args.vararg.arg)
    if func_def.args.kwarg:
        params.add(func_def.args.kwarg.arg)

    # Collect local assignments (variables defined in function)
    locals_assigned = set()
    for node in ast.walk(func_def):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            locals_assigned.add(node.id)
        elif isinstance(node, ast.For):
            if isinstance(node.target, ast.Name):
                locals_assigned.add(node.target.id)
            elif isinstance(node.target, ast.Tuple):
                for elt in node.target.elts:
                    if isinstance(elt, ast.Name):
                        locals_assigned.add(elt.id)
        elif isinstance(node, ast.comprehension) and isinstance(node.target, ast.Name):
            locals_assigned.add(node.target.id)

    # Collect imports
    imports = set()
    for node in ast.walk(func_def):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports.add(alias.asname or alias.name)

    # Get builtins
    builtin_names = set(dir(builtins))

    # Find all Name nodes that are loads (reads)
    potential_globals = []
    for node in ast.walk(func_def):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            name = node.id
            # Skip if it's a parameter, local, import, or builtin
            if (
                name not in params
                and name not in locals_assigned
                and name not in imports
                and name not in builtin_names
                and name not in potential_globals
            ):
                potential_globals.append(name)

    return potential_globals


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
    """Update a Twin's private side with the computation result.

    When the result is itself a Twin (from approve()), we merge its properties
    into the existing Twin rather than nesting. This way:
    - existing.private gets the received Twin's private value
    - existing.private_stdout/stderr/figures get copied over
    - existing.public stays unchanged (user already had mock data)
    """
    if comp_id in _TWIN_RESULTS:
        existing_twin = _TWIN_RESULTS[comp_id]

        # Check if received value is a Twin - if so, merge its properties
        from .twin import Twin, _sync_twin_instances

        if isinstance(result_value, Twin):
            # Merge received Twin's properties into existing Twin
            # The received Twin has public=None, private=actual_result
            object.__setattr__(existing_twin, "private", result_value.private)

            # Copy over captured outputs from the received Twin
            for attr in ("private_stdout", "private_stderr", "private_figures"):
                if hasattr(result_value, attr):
                    val = getattr(result_value, attr)
                    if val is not None:
                        setattr(existing_twin, attr, val)

            print(
                f"✨ Twin result auto-updated: {existing_twin.name or existing_twin.twin_id[:8]}..."
            )
            print("   .private now contains the approved result")
            print("   .public still has your local mock data")
        else:
            # Non-Twin result - just set as private (legacy/simple case)
            object.__setattr__(existing_twin, "private", result_value)
            print(
                f"✨ Twin result auto-updated: {existing_twin.name or existing_twin.twin_id[:8]}..."
            )
            print("   .private now contains the approved result")

        # Sync all Twin instances with the same twin_id to point to the canonical object
        _sync_twin_instances(existing_twin.twin_id, existing_twin.owner, existing_twin)

        # Remove from registry after updating
        del _TWIN_RESULTS[comp_id]
        return True
    return False


def _describe_bound_data(args, kwargs, context=None) -> list[str]:
    """
    Generate human-readable lines describing bound Twin/RemoteVar arguments.

    Mirrors the display used in ComputationRequest.__repr__ so other callers
    (e.g., envelope summaries) can reuse the same snapshot of bound data.
    """
    lines = []
    for _i, arg in enumerate(args):
        from .twin import Twin

        if isinstance(arg, Twin):
            underlying_type = (
                arg.var_type.replace("Twin[", "").replace("]", "")
                if arg.var_type.startswith("Twin[")
                else arg.var_type
            )
            privacy_badge = (
                "⚠️  REAL+MOCK"
                if arg.has_private and arg.has_public
                else "🔒 PRIVATE"
                if arg.has_private
                else "🌍 PUBLIC"
                if arg.has_public
                else "⏳ PENDING"
            )
            lines.append(f"  │ {privacy_badge}")
            lines.append(f"  │   Parameter: {arg.name or 'unnamed'}")
            lines.append(f"  │   Type: {underlying_type}")
            lines.append(f"  │   Owner: {arg.owner}")
            if arg.has_public:
                lines.append("  │   📊 Mock data available for testing")
            if arg.has_private:
                owner = arg.owner
                lines.append(f"  │   🔐 Real data available (owner: {owner})")

        elif isinstance(arg, dict) and arg.get("_beaver_remote_var"):
            var_type = arg.get("var_type", "unknown")
            underlying_type = (
                var_type.replace("Twin[", "").replace("]", "")
                if var_type.startswith("Twin[")
                else var_type
            )
            badge = "🔗 TWIN REF" if var_type.startswith("Twin[") else "🔗 DATA REF"
            owner = arg.get("owner", "unknown")
            lines.append(f"  │ {badge}")
            lines.append(f"  │   Parameter: {arg.get('name', 'unnamed')}")
            lines.append(f"  │   Type: {underlying_type}")
            lines.append(f"  │   Owner: {owner}")

    for _k, v in kwargs.items():
        # Reuse the same logic for kwargs
        lines.extend(_describe_bound_data((v,), {}, context))

    return lines


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
    result_id: Optional[str] = None
    context: Any = None
    session: Any = None  # Session reference for session-scoped results

    def __post_init__(self):
        # Ensure result_id always set; default to var_id if not provided
        if not self.result_id:
            self.result_id = self.var_id

    @property
    def data(self):
        """Get the result data."""
        return self.result

    @data.setter
    def data(self, value):
        """Modify the result data before approval."""
        self.result = value

    # Accessors for captured outputs from the result Twin
    # Use object.__getattribute__ to bypass Twin's __getattr__ magic
    @property
    def public_stdout(self) -> Optional[str]:
        """Get captured stdout from public/mock execution."""
        if self.result is None:
            return None
        try:
            return object.__getattribute__(self.result, "public_stdout")
        except AttributeError:
            return None

    @property
    def public_stderr(self) -> Optional[str]:
        """Get captured stderr from public/mock execution."""
        if self.result is None:
            return None
        try:
            return object.__getattribute__(self.result, "public_stderr")
        except AttributeError:
            return None

    @property
    def public_figures(self) -> Optional[list]:
        """Get captured figures from public/mock execution."""
        if self.result is None:
            return None
        try:
            return object.__getattribute__(self.result, "public_figures")
        except AttributeError:
            return None

    @property
    def private_stdout(self) -> Optional[str]:
        """Get captured stdout from private/real execution."""
        if self.result is None:
            return None
        try:
            return object.__getattribute__(self.result, "private_stdout")
        except AttributeError:
            return None

    @property
    def private_stderr(self) -> Optional[str]:
        """Get captured stderr from private/real execution."""
        if self.result is None:
            return None
        try:
            return object.__getattribute__(self.result, "private_stderr")
        except AttributeError:
            return None

    @property
    def private_figures(self) -> Optional[list]:
        """Get captured figures from private/real execution."""
        if self.result is None:
            return None
        try:
            return object.__getattribute__(self.result, "private_figures")
        except AttributeError:
            return None

    def show_figures(self, which: str = "both"):
        """Display captured figures. which: 'public', 'private', or 'both'."""
        if self.result and hasattr(self.result, "show_figures"):
            self.result.show_figures(which)
        else:
            print("📊 No figures available")

    def show_output(self, which: str = "both"):
        """Display captured stdout/stderr. which: 'public', 'private', or 'both'."""
        if self.result and hasattr(self.result, "show_output"):
            self.result.show_output(which)
        else:
            # Fall back to execution stdout/stderr
            if self.stdout:
                print(f"📤 stdout: {self.stdout}")
            if self.stderr:
                print(f"⚠️  stderr: {self.stderr}")

    def approve(
        self,
        *,
        stdout: Optional[str] = ...,  # ... means "use captured value"
        stderr: Optional[str] = ...,
        include_output: bool = True,
    ):
        """
        Approve and send the private/real result back to the requester.

        Args:
            stdout: Override stdout to send (None to exclude, ... to use captured)
            stderr: Override stderr to send (None to exclude, ... to use captured)
            include_output: If False, don't send any stdout/stderr

        If result is a Twin (from run_both), extracts just the private value to send.
        The requester already has access to mock/public data, so we only send private.
        Captured outputs (stdout, stderr, figures) are included separately.
        """
        from .twin import Twin

        if isinstance(self.result, Twin):
            if self.result.has_private:
                print(
                    f"✅ Approving result for: {self.var_name} "
                    f"(twin_id={self.result.twin_id[:12]}..., comp_id={self.comp_id[:12]}..., result_id={self.result_id[:12]})"
                )
                # Extract just the private value - don't send the whole Twin
                # The requester already ran mock locally, they just need the private result
                private_value = self.result.private

                # Get captured outputs from the Twin to pass along
                captured_stdout = stdout
                captured_stderr = stderr
                captured_figures = None

                if stdout is ...:
                    captured_stdout = getattr(self.result, "private_stdout", None)
                if stderr is ...:
                    captured_stderr = getattr(self.result, "private_stderr", None)
                captured_figures = getattr(self.result, "private_figures", None)

                return self._send_result(
                    private_value,
                    stdout=captured_stdout,
                    stderr=captured_stderr,
                    include_output=include_output,
                    captured_figures=captured_figures,
                )
            else:
                raise ValueError("Twin has no private data to approve")
        else:
            return self._send_result(
                self.result,
                stdout=stdout,
                stderr=stderr,
                include_output=include_output,
            )

    def approve_mock(self):
        """
        Approve and send only the mock/public result back.

        Useful for iterative development - lets requester continue working
        with mock data while you review the private computation.
        """
        from .twin import Twin

        if isinstance(self.result, Twin):
            if self.result.has_public:
                print(
                    f"🧪 Approving mock/public result for: {self.var_name} "
                    f"(twin_id={self.result.twin_id[:12]}..., comp_id={self.comp_id[:12]}..., result_id={self.result_id[:12]})"
                )
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

        # Send rejection back - use session data folder if available
        if self.session is not None:
            from pathlib import Path

            from .runtime import pack, write_envelope

            # Use data/ subdirectory for all beaver files
            data_dir = Path(self.session.local_folder) / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            env = pack(
                rejection,
                sender=self.context.user,
                name=f"rejection_{self.var_name}",
                reply_to=self.comp_id,
            )
            result = write_envelope(env, out_dir=data_dir)
            print(f"✓ Rejection sent to session data folder: {data_dir}")
        else:
            result = self.context.send(
                rejection,
                name=f"rejection_{self.var_name}",
                user=self.sender,
                reply_to=self.comp_id,
            )
            print(f"✓ Rejection sent to {self.sender}'s inbox")
        return result

    def _send_result(
        self,
        value,
        *,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
        include_output: bool = True,
        captured_figures: Optional[list] = None,
    ):
        """Send the result back to the requester.

        Args:
            value: The value to send (raw result, not a Twin)
            stdout: Stdout to include with result
            stderr: Stderr to include with result
            include_output: If False, don't send any stdout/stderr
            captured_figures: List of CapturedFigure objects to include
        """
        if not self.context:
            raise ValueError("No context available for sending result")

        print(f"   Sending to: {self.sender}")

        # Convert matplotlib Figure to PNG bytes for serialization
        def fig_to_png_bytes(fig):
            """Convert matplotlib Figure to PNG bytes."""
            import io

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            buf.seek(0)
            return buf.getvalue()

        def make_serializable(obj):
            """Recursively convert non-serializable types to serializable ones."""
            # Handle None early
            if obj is None:
                return None

            # Handle matplotlib Figure
            try:
                import matplotlib.figure

                if isinstance(obj, matplotlib.figure.Figure):
                    return {"_beaver_figure": True, "png_bytes": fig_to_png_bytes(obj)}
            except ImportError:
                pass

            # Handle matplotlib Axes - convert to figure PNG
            try:
                from matplotlib.axes import Axes

                if isinstance(obj, Axes):
                    if obj.figure:
                        return {"_beaver_figure": True, "png_bytes": fig_to_png_bytes(obj.figure)}
                    return {"_beaver_axes": True, "title": str(obj.get_title())}
            except ImportError:
                pass

            # Handle AnnData - convert to a serializable summary
            try:
                import anndata

                if isinstance(obj, anndata.AnnData):
                    # Check if there's a TrustedLoader registered for AnnData
                    from .runtime import TrustedLoader

                    tl = TrustedLoader.get(type(obj))
                    if tl:
                        # Use TrustedLoader - will be handled by _prepare_for_sending
                        return obj
                    # Otherwise convert to dict summary (lossy but serializable)
                    return {
                        "_anndata_summary": True,
                        "n_obs": obj.n_obs,
                        "n_vars": obj.n_vars,
                        "obs_names": list(obj.obs_names[:10]) + (["..."] if obj.n_obs > 10 else []),
                        "var_names": list(obj.var_names[:10])
                        + (["..."] if obj.n_vars > 10 else []),
                        "shape": obj.shape,
                    }
            except ImportError:
                pass

            # Handle scipy sparse matrices
            try:
                import scipy.sparse

                if scipy.sparse.issparse(obj):
                    # Convert to dense if small, otherwise to COO format data
                    if obj.nnz < 10000:
                        return obj.toarray().tolist()
                    return {
                        "_sparse_matrix": True,
                        "shape": obj.shape,
                        "nnz": obj.nnz,
                        "format": obj.format,
                        "data": obj.data.tolist()[:1000],  # First 1000 values
                        "note": "Truncated for serialization",
                    }
            except ImportError:
                pass

            # Handle pandas DataFrames and Series
            try:
                import pandas as pd

                if isinstance(obj, pd.DataFrame):
                    return obj.to_dict(orient="list")
                if isinstance(obj, pd.Series):
                    return obj.to_dict()
            except ImportError:
                pass

            # Handle numpy types
            try:
                import numpy as np

                if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                    return int(obj)
                if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                    return float(obj)
                if isinstance(obj, np.bool_):
                    return bool(obj)
                if isinstance(obj, np.ndarray):
                    # For large arrays, truncate
                    if obj.size > 10000:
                        return {
                            "_numpy_array": True,
                            "shape": obj.shape,
                            "dtype": str(obj.dtype),
                            "sample": obj.flatten()[:100].tolist(),
                            "note": "Truncated for serialization",
                        }
                    return obj.tolist()
            except ImportError:
                pass

            # Handle collections recursively
            if isinstance(obj, tuple):
                return tuple(make_serializable(x) for x in obj)
            if isinstance(obj, list):
                return [make_serializable(x) for x in obj]
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}

            # Handle objects that might not be serializable
            # Check if it's a basic type that pyfory can handle
            if isinstance(obj, (str, int, float, bool, bytes)):
                return obj

            # For unknown objects, try to convert to a safe representation
            obj_type = type(obj)
            type_name = f"{obj_type.__module__}.{obj_type.__name__}"

            # Catch any matplotlib objects we missed
            if "matplotlib" in type_name:
                try:
                    return {"_matplotlib_object": True, "type": type_name, "repr": repr(obj)[:200]}
                except Exception:
                    return {"_matplotlib_object": True, "type": type_name}

            # Check if object lacks serializable state
            if not hasattr(obj, "__dict__") and not hasattr(obj, "__slots__"):
                # Object without state - return a placeholder
                return {"_unserializable": True, "type": type_name, "repr": repr(obj)[:200]}

            # Try to detect other problematic objects by checking if they're from
            # common scientific libraries that have complex internal state
            problematic_modules = ("scanpy", "anndata", "matplotlib", "scipy", "sklearn")
            if any(mod in type_name for mod in problematic_modules):
                try:
                    return {"_complex_object": True, "type": type_name, "repr": repr(obj)[:200]}
                except Exception:
                    return {"_complex_object": True, "type": type_name}

            return obj

        def convert_figures_for_sending(figs):
            """Convert list of CapturedFigure/Figure to serializable dicts."""
            if not figs:
                return []
            result = []
            for fig_item in figs:
                if hasattr(fig_item, "png_bytes"):
                    # CapturedFigure - already has PNG bytes
                    result.append({"_beaver_figure": True, "png_bytes": fig_item.png_bytes})
                elif hasattr(fig_item, "savefig"):
                    # Raw matplotlib Figure
                    result.append({"_beaver_figure": True, "png_bytes": fig_to_png_bytes(fig_item)})
            return result

        # Make the value serializable (convert Figures, numpy types, etc.)
        value_to_send = make_serializable(value)

        # Handle None results - use sentinel dict to allow Twin creation
        # Functions that return None (like plotting functions) are valid
        if value_to_send is None:
            value_to_send = {"_none_result": True, "has_figures": bool(captured_figures)}

        # Create a Twin with private=result so receiver gets proper structure
        # They already have public/mock data, so we only populate private side
        from .twin import Twin

        result_twin = Twin(
            public=None,  # They already have this from their local mock run
            private=value_to_send,
            owner=self.context.user,
            name=self.var_name,
            twin_id=self.result_id,
        )

        # Attach captured outputs to the Twin
        if include_output:
            if stdout is not None:
                result_twin.private_stdout = stdout
            if stderr is not None:
                result_twin.private_stderr = stderr
            if captured_figures:
                result_twin.private_figures = convert_figures_for_sending(captured_figures)

        # Send the result Twin back as a reply
        # Use session folder if available, otherwise fall back to user inbox
        if self.session is not None:
            # Write to our session data folder (peer can read it via sync)
            from pathlib import Path

            from .runtime import pack, write_envelope

            # Use data/ subdirectory for all beaver files
            data_dir = Path(self.session.local_folder) / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            # Determine backend and recipients for encryption
            # In SyftBox, data is encrypted FOR THE RECIPIENT (peer) only
            backend = None
            recipients = None
            if self.context and hasattr(self.context, "_backend") and self.context._backend:
                backend = self.context._backend
                if backend.uses_crypto:
                    # Encrypt for the sender who requested the computation
                    recipients = [self.sender]

            # Preserve private side when wrapping in Twin so it survives serialization
            preserve_private = isinstance(result_twin, Twin) and result_twin.has_private

            env = pack(
                result_twin,
                sender=self.context.user,
                name=self.var_name,
                reply_to=self.comp_id,
                artifact_dir=data_dir,
                backend=backend,
                recipients=recipients,
                preserve_private=preserve_private,
            )

            # Write envelope (encrypted if backend available)
            if backend and backend.uses_crypto and recipients:
                import base64
                import json

                record = {
                    "version": env.version,
                    "envelope_id": env.envelope_id,
                    "sender": env.sender,
                    "created_at": env.created_at,
                    "name": env.name,
                    "inputs": env.inputs,
                    "outputs": env.outputs,
                    "requirements": env.requirements,
                    "manifest": env.manifest,
                    "reply_to": env.reply_to,
                    "payload_b64": base64.b64encode(env.payload).decode("ascii"),
                }
                content = json.dumps(record, indent=2).encode("utf-8")
                data_path = data_dir / env.filename()
                backend.storage.write_with_shadow(
                    str(data_path),
                    content,
                    recipients=recipients,
                    hint="beaver-envelope",
                )
                result = data_path
            else:
                result = write_envelope(env, out_dir=data_dir)
            print(f"✓ Result sent to session data folder: {data_dir}")
        else:
            # Fallback: send to user's inbox (legacy path)
            result = self.context.send(
                result_twin,
                name=self.var_name,
                user=self.sender,
                reply_to=self.comp_id,
                preserve_private=True,
            )
            print(f"✓ Result sent to {self.sender}'s inbox")
        print(f"💡 They can load it with: bv.inbox()['{self.var_name}'].load()")

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
    # Required package versions detected from sender's environment
    required_versions: dict = field(default_factory=dict)

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
        # Pass session reference if we have one
        session = getattr(self, "_session", None)
        if session is None and context is not None:
            session = getattr(context, "_active_session", None)
            if session is not None:
                self._session = session
        return ComputationResult(
            result=result_data["result"],
            stdout=result_data["stdout"],
            stderr=result_data["stderr"],
            error=result_data["error"],
            var_name=self.result_name,
            var_id=self.result_id,
            result_id=self.result_id,
            comp_id=self.comp_id,
            sender=self.sender,
            context=context,
            session=session,
        )

    def run(self, context=None):
        """
        Execute the computation on real/private data only.

        Returns ComputationResult where .data is a Twin with only private side.
        If execution fails, returns ComputationResult with error (no Twin).
        """
        from .remote_vars import _ensure_sparse_shapes, _SafeDisplayProxy
        from .twin import _TWIN_REGISTRY, CapturedFigure, Twin

        def unwrap_for_computation(val):
            """Unwrap display proxies and ensure sparse shapes for computation."""
            if isinstance(val, _SafeDisplayProxy):
                val = val._raw
            return _ensure_sparse_shapes(val)

        def get_local_twin_private(arg):
            """Look up local Twin and get its private data (preferring local over received)."""

            def _get_private_data(twin):
                """Get private data, auto-loading TrustedLoader if needed."""
                raw_private = twin._get_raw_private()
                if raw_private is None:
                    return None
                # Check if it's a TrustedLoader that needs resolution
                if twin._is_trusted_loader(raw_private):
                    # Auto-load since owner is running the computation (implicit consent)
                    print(f"🔓 Auto-loading private data for '{twin.name}'...")
                    loaded = twin.load(which="private", auto_accept=False)
                    if loaded is None:
                        return None
                    return unwrap_for_computation(loaded)
                return unwrap_for_computation(raw_private)

            if context:
                twin_id = arg.twin_id
                owner = arg.owner

                # Check global Twin registry for the executing user's version
                key = (twin_id, context.user)
                if key in _TWIN_REGISTRY:
                    local_twin = _TWIN_REGISTRY[key]
                    private_data = _get_private_data(local_twin)
                    if private_data is not None:
                        return private_data

                # Also check with arg's owner
                if owner != context.user:
                    key = (twin_id, owner)
                    if key in _TWIN_REGISTRY:
                        local_twin = _TWIN_REGISTRY[key]
                        private_data = _get_private_data(local_twin)
                        if private_data is not None:
                            return private_data

                # Fallback: any Twin with matching id
                for (tid, _owner), local_twin in _TWIN_REGISTRY.items():
                    if tid == twin_id:
                        private_data = _get_private_data(local_twin)
                        if private_data is not None:
                            return private_data

            # Fall back to received Twin's private
            private_data = _get_private_data(arg)
            if private_data is not None:
                return private_data
            return None

        # Auto-detect context if needed
        if context is None:
            context = self._auto_detect_context()

        def resolve_twin_ref(val):
            """Convert Twin reference dicts back into local Twin instances if available."""
            if isinstance(val, dict) and val.get("_beaver_twin_ref"):
                from .twin import _TWIN_REGISTRY, Twin

                twin_id = val["twin_id"]
                owner_hint = val.get("owner")
                dataset_syft_url = val.get("syft_url")
                dataset_asset = val.get("dataset_asset")
                if owner_hint and (twin_id, owner_hint) in _TWIN_REGISTRY:
                    return _TWIN_REGISTRY[(twin_id, owner_hint)]
                if context and (twin_id, context.user) in _TWIN_REGISTRY:
                    return _TWIN_REGISTRY[(twin_id, context.user)]
                # Fallback: search any registered Twin by id
                for (tid, _owner), twin in _TWIN_REGISTRY.items():
                    if tid == twin_id:
                        return twin
                # Last chance: search context.remote_vars stored values
                if context and hasattr(context, "remote_vars"):
                    for var in context.remote_vars.vars.values():
                        sv = getattr(var, "_stored_value", None)
                        if isinstance(sv, Twin) and sv.twin_id == twin_id:
                            # Register for future lookups
                            _TWIN_REGISTRY[(sv.twin_id, sv.owner)] = sv
                            return sv
                # Also try matching by name in session's remote_vars (for DO receiving requests about their own Twins)
                twin_name = val.get("name")
                if (
                    twin_name
                    and context
                    and hasattr(context, "_active_session")
                    and context._active_session
                ):
                    session = context._active_session
                    if hasattr(session, "remote_vars") and twin_name in session.remote_vars.vars:
                        var = session.remote_vars.vars[twin_name]
                        # Check _twin_reference first (set when publishing Twin), then _stored_value
                        sv = getattr(var, "_twin_reference", None) or getattr(
                            var, "_stored_value", None
                        )
                        if isinstance(sv, Twin):
                            _TWIN_REGISTRY[(sv.twin_id, sv.owner)] = sv
                            return sv
                # Try loading from datasets if available (owner publishes assets)
                if context and hasattr(context, "datasets"):
                    try:
                        # Prefer explicit dataset URL + asset key if provided
                        twin = None
                        if dataset_syft_url:
                            dataset = context.datasets.load_from_url(dataset_syft_url)
                            if dataset_asset:
                                twin = dataset[dataset_asset]
                            else:
                                twin = dataset.get_asset_by_id(twin_id)
                        else:
                            ds_owner = owner_hint or context.user
                            twin = context.datasets.find_by_twin_id(ds_owner, twin_id)
                        if twin:
                            _TWIN_REGISTRY[(twin.twin_id, twin.owner)] = twin
                            print(
                                f"📦 Loaded dataset Twin '{twin.name}' "
                                f"(owner={twin.owner}, id={twin_id[:8]}...)"
                            )
                            return twin
                    except Exception:
                        # Avoid breaking computation if dataset lookup fails
                        pass
                # Debug: show what we searched
                debug_info = []
                debug_info.append(
                    f"twin_id={twin_id[:12]}, name={val.get('name')}, owner_hint={owner_hint}"
                )
                debug_info.append(f"_TWIN_REGISTRY keys: {list(_TWIN_REGISTRY.keys())[:5]}")
                if context:
                    debug_info.append(f"context.user={getattr(context, 'user', None)}")
                    if hasattr(context, "_active_session") and context._active_session:
                        sess = context._active_session
                        debug_info.append(
                            f"session exists, remote_vars.vars keys: {list(sess.remote_vars.vars.keys()) if hasattr(sess, 'remote_vars') else 'N/A'}"
                        )
                    else:
                        debug_info.append("_active_session is None or missing")
                else:
                    debug_info.append("context is None")
                raise ValueError(
                    f"Twin reference '{val.get('name', 'unknown')}' (ID: {twin_id[:12]}...) not available locally\n"
                    + "\n".join(debug_info)
                )
            return val

        resolved_args = tuple(resolve_twin_ref(a) for a in self.args)
        resolved_kwargs = {k: resolve_twin_ref(v) for k, v in self.kwargs.items()}

        # Also resolve function references (TwinFunc callbacks)
        from .twin_func import resolve_func_ref

        resolved_args = tuple(resolve_func_ref(a, context) for a in resolved_args)
        resolved_kwargs = {k: resolve_func_ref(v, context) for k, v in resolved_kwargs.items()}

        # Replace Twin arguments with their private sides (preferring local Twin)
        private_args = []
        for arg in resolved_args:
            if isinstance(arg, Twin):
                private_data = get_local_twin_private(arg)
                if private_data is not None:
                    private_args.append(private_data)
                else:
                    # Give detailed error about why private data isn't available
                    raw_priv = arg._get_raw_private()
                    if raw_priv is None:
                        hint = "Twin has no private slot set."
                        # Check if there's a mapping that should exist
                        if hasattr(arg, "syft_url") and arg.syft_url:
                            hint += f"\n   syft_url: {arg.syft_url}"
                        if hasattr(arg, "dataset_asset") and arg.dataset_asset:
                            hint += f"\n   dataset_asset: {arg.dataset_asset}"
                        hint += "\n   💡 Check that mapping.yaml has an entry for the private URL"
                    elif arg._is_trusted_loader(raw_priv):
                        loader_path = raw_priv.get("path", "unknown")
                        hint = "Twin has TrustedLoader but load failed."
                        hint += f"\n   Loader path: {loader_path}"
                        # Check if file exists
                        from pathlib import Path

                        if not Path(loader_path).exists():
                            hint += "\n   ❌ File does NOT exist at this path!"
                            hint += "\n   💡 Check mapping.yaml entry maps to the correct file"
                    else:
                        hint = f"Twin private contains unexpected type: {type(raw_priv)}"
                    raise ValueError(
                        f"Twin '{arg.name}' (id={arg.twin_id[:8]}...) has no private data.\n   {hint}"
                    )
            else:
                private_args.append(arg)

        private_kwargs = {}
        for key, val in resolved_kwargs.items():
            if isinstance(val, Twin):
                private_data = get_local_twin_private(val)
                if private_data is not None:
                    private_kwargs[key] = private_data
                else:
                    raise ValueError(f"Twin kwarg '{key}' has no private data for real execution")
            else:
                private_kwargs[key] = val

        # Set up output capture
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        captured_figures = []
        captured_fig_ids = set()

        def _capture_figure(fig):
            """Save a matplotlib figure to PNG once per figure id."""
            nonlocal captured_figures, captured_fig_ids
            if fig is None:
                return
            if id(fig) in captured_fig_ids:
                return
            captured_fig_ids.add(id(fig))
            buf = io.BytesIO()
            try:
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
                buf.seek(0)
                captured_figures.append(
                    CapturedFigure(
                        {
                            "figure": None,
                            "png_bytes": buf.getvalue(),
                        }
                    )
                )
            except Exception:
                pass

        # Try to capture matplotlib figures
        try:
            import matplotlib
            import matplotlib.pyplot as plt

            # Close ALL existing figures to ensure clean state
            plt.close("all")
            original_backend = matplotlib.get_backend()
            with suppress(Exception):
                matplotlib.use("Agg", force=True)
            has_matplotlib = True

            # Hook plt.show() to capture figures BEFORE they're potentially closed
            original_show = plt.show

            def capturing_show(*_args, **_kwargs):
                """Capture all current figures when show() is called."""
                for fig_num in plt.get_fignums():
                    fig = plt.figure(fig_num)
                    _capture_figure(fig)

            plt.show = capturing_show

        except ImportError:
            has_matplotlib = False

        error = None
        private_result = None

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                private_result = self.func(*private_args, **private_kwargs)

            # Capture any remaining matplotlib figures that weren't captured by show()
            if has_matplotlib:
                # Restore original show
                plt.show = original_show

                # Collect figures to capture: all figures AND figures from returned Axes
                figures_to_capture = set()

                # Add all remaining figures (we closed all before, so all are new)
                for fig_num in plt.get_fignums():
                    figures_to_capture.add(plt.figure(fig_num))

                # Also check if result contains Axes - if so, capture their figures
                def extract_axes_figures(obj):
                    """Extract parent figures from Axes objects in result."""
                    figs = set()
                    try:
                        from matplotlib.axes import Axes

                        if isinstance(obj, Axes):
                            if obj.figure:
                                figs.add(obj.figure)
                        elif isinstance(obj, (list, tuple)):
                            for item in obj:
                                figs.update(extract_axes_figures(item))
                        elif isinstance(obj, dict):
                            for item in obj.values():
                                figs.update(extract_axes_figures(item))
                    except Exception:
                        pass
                    return figs

                figures_to_capture.update(extract_axes_figures(private_result))

                # Capture all unique figures
                for fig in figures_to_capture:
                    _capture_figure(fig)
                    plt.close(fig)

                # Clean up and restore original backend
                plt.close("all")
                with suppress(Exception):
                    matplotlib.use(original_backend, force=True)

        except Exception as e:
            error = str(e)
            import traceback

            stderr_capture.write(traceback.format_exc())
            # Restore matplotlib state on error
            if has_matplotlib:
                plt.show = original_show
                plt.close("all")
                with suppress(Exception):
                    matplotlib.use(original_backend, force=True)

        stdout_str = stdout_capture.getvalue()
        stderr_str = stderr_capture.getvalue()

        # Handle None result (e.g., plotting functions)
        if private_result is None and error is None:
            private_result = {"_none_result": True, "has_figures": bool(captured_figures)}

        # Create result Twin with figures
        if error is None:
            result_twin = Twin(
                private=private_result,
                public=None,
                owner=context.user if context else "unknown",
                name=self.result_name,
                twin_id=self.result_id,
            )
            result_twin.private_stdout = stdout_str
            result_twin.private_stderr = stderr_str
            result_twin.private_figures = captured_figures
        else:
            result_twin = None

        # Store for later retrieval after approval
        _TWIN_RESULTS[self.comp_id] = result_twin

        # Create ComputationResult with session reference if we have one
        session = getattr(self, "_session", None)
        if session is None and context is not None:
            session = getattr(context, "_active_session", None)
            if session is not None:
                self._session = session
        comp_result = ComputationResult(
            result=result_twin,
            stdout=stdout_str,
            stderr=stderr_str,
            error=error,
            var_name=self.result_name,
            var_id=self.result_id,
            result_id=self.result_id,
            comp_id=self.comp_id,
            sender=self.sender,
            context=context,
            session=session,
        )

        print(f"⚙️  Executed: {self.result_name} = {self.func.__name__}(...)")
        print(f"   From: {self.sender}")
        if context:
            print(f"   Context: {context.user}")
        if captured_figures:
            print(f"   Captured {len(captured_figures)} figure(s)")
        if error:
            print(f"❌ Error: {error}")
        else:
            print("✓ Execution complete")

        return comp_result

    def run_mock(self, context=None):
        """
        Execute the computation on mock/public data for safe preview.

        Returns ComputationResult where .data is a Twin with only public side.
        """
        from .remote_vars import _ensure_sparse_shapes, _SafeDisplayProxy
        from .twin import _TWIN_REGISTRY, Twin

        def unwrap_for_computation(val):
            """Unwrap display proxies and ensure sparse shapes for computation."""
            # Unwrap SafeDisplayProxy to get raw object
            if isinstance(val, _SafeDisplayProxy):
                val = val._raw
            # Ensure sparse matrices have _shape attribute
            return _ensure_sparse_shapes(val)

        def get_local_twin_public(arg):
            """Look up local Twin and get its public data (preferring local over received)."""

            def _get_public_data(twin):
                """Get public data, auto-loading TrustedLoader if needed."""
                raw_public = twin._get_raw_public()
                if raw_public is None:
                    return None
                # Check if it's a TrustedLoader that needs resolution
                if twin._is_trusted_loader(raw_public):
                    # Auto-load since this is for mock testing
                    print(f"🌍 Auto-loading public data for '{twin.name}'...")
                    loaded = twin.load(which="public", auto_accept=False)
                    if loaded is None:
                        return None
                    return unwrap_for_computation(loaded)
                return unwrap_for_computation(raw_public)

            if context:
                # Try to find the owner's local version of this Twin
                twin_id = arg.twin_id
                owner = arg.owner

                # Check global Twin registry for the executing user's version
                key = (twin_id, context.user)
                if key in _TWIN_REGISTRY:
                    local_twin = _TWIN_REGISTRY[key]
                    public_data = _get_public_data(local_twin)
                    if public_data is not None:
                        return public_data

                # Also check with arg's owner
                if owner != context.user:
                    key = (twin_id, owner)
                    if key in _TWIN_REGISTRY:
                        local_twin = _TWIN_REGISTRY[key]
                        public_data = _get_public_data(local_twin)
                        if public_data is not None:
                            return public_data

                # Fallback: any Twin with matching id
                for (tid, _owner), local_twin in _TWIN_REGISTRY.items():
                    if tid == twin_id:
                        public_data = _get_public_data(local_twin)
                        if public_data is not None:
                            return public_data

            # Fall back to received Twin's public
            public_data = _get_public_data(arg)
            if public_data is not None:
                return public_data
            return None

        # Auto-detect context if needed
        if context is None:
            context = self._auto_detect_context()

        def resolve_twin_ref(val):
            """Convert Twin reference dicts back into local Twin instances if available."""
            if isinstance(val, dict) and val.get("_beaver_twin_ref"):
                from .twin import _TWIN_REGISTRY, Twin

                twin_id = val["twin_id"]
                owner_hint = val.get("owner")
                dataset_syft_url = val.get("syft_url")
                dataset_asset = val.get("dataset_asset")
                # Try owner hint
                if owner_hint and (twin_id, owner_hint) in _TWIN_REGISTRY:
                    return _TWIN_REGISTRY[(twin_id, owner_hint)]
                # Try context user
                if context and (twin_id, context.user) in _TWIN_REGISTRY:
                    return _TWIN_REGISTRY[(twin_id, context.user)]
                # Fallback: any twin with matching id
                for (tid, _owner), twin in _TWIN_REGISTRY.items():
                    if tid == twin_id:
                        return twin
                # Last chance: search context.remote_vars stored values
                if context and hasattr(context, "remote_vars"):
                    for var in context.remote_vars.vars.values():
                        sv = getattr(var, "_stored_value", None)
                        if isinstance(sv, Twin) and sv.twin_id == twin_id:
                            _TWIN_REGISTRY[(sv.twin_id, sv.owner)] = sv
                            return sv
                # Also try matching by name in session's remote_vars (for DO receiving requests about their own Twins)
                twin_name = val.get("name")
                if (
                    twin_name
                    and context
                    and hasattr(context, "_active_session")
                    and context._active_session
                ):
                    session = context._active_session
                    if hasattr(session, "remote_vars") and twin_name in session.remote_vars.vars:
                        var = session.remote_vars.vars[twin_name]
                        # Check _twin_reference first (set when publishing Twin), then _stored_value
                        sv = getattr(var, "_twin_reference", None) or getattr(
                            var, "_stored_value", None
                        )
                        if isinstance(sv, Twin):
                            _TWIN_REGISTRY[(sv.twin_id, sv.owner)] = sv
                            return sv
                # Try loading from datasets if available (owner publishes assets)
                if context and hasattr(context, "datasets"):
                    try:
                        # Prefer explicit dataset URL + asset key if provided
                        twin = None
                        if dataset_syft_url:
                            dataset = context.datasets.load_from_url(dataset_syft_url)
                            if dataset_asset:
                                twin = dataset[dataset_asset]
                            else:
                                twin = dataset.get_asset_by_id(twin_id)
                        else:
                            ds_owner = owner_hint or context.user
                            twin = context.datasets.find_by_twin_id(ds_owner, twin_id)
                        if twin:
                            _TWIN_REGISTRY[(twin.twin_id, twin.owner)] = twin
                            print(
                                f"📦 Loaded dataset Twin '{twin.name}' "
                                f"(owner={twin.owner}, id={twin_id[:8]}...)"
                            )
                            return twin
                    except Exception:
                        # Avoid breaking computation if dataset lookup fails
                        pass
                raise ValueError(
                    f"Twin reference '{val.get('name', 'unknown')}' (ID: {twin_id[:12]}...) not available locally"
                )
            return val

        resolved_args = tuple(resolve_twin_ref(a) for a in self.args)
        resolved_kwargs = {k: resolve_twin_ref(v) for k, v in self.kwargs.items()}

        # Also resolve function references (TwinFunc callbacks)
        from .twin_func import resolve_func_ref

        resolved_args = tuple(resolve_func_ref(a, context) for a in resolved_args)
        resolved_kwargs = {k: resolve_func_ref(v, context) for k, v in resolved_kwargs.items()}

        # Replace Twin arguments with their public sides (preferring local Twin)
        mock_args = []
        for arg in resolved_args:
            if isinstance(arg, Twin):
                public_data = get_local_twin_public(arg)
                if public_data is not None:
                    mock_args.append(public_data)
                else:
                    raise ValueError("Twin argument has no public data for mock testing")
            else:
                mock_args.append(arg)

        mock_kwargs = {}
        for k, v in resolved_kwargs.items():
            if isinstance(v, Twin):
                public_data = get_local_twin_public(v)
                if public_data is not None:
                    mock_kwargs[k] = public_data
                else:
                    raise ValueError(f"Twin kwarg '{k}' has no public data for mock testing")
            else:
                mock_kwargs[k] = v

        # Execute on mock data with output capture
        print("🧪 Testing on mock/public data...")

        # Capture stdout, stderr, and matplotlib figures
        from .twin import CapturedFigure

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        captured_figures = []
        captured_fig_ids = set()  # Track figure IDs to prevent duplicates

        # Try to capture matplotlib figures
        try:
            import matplotlib
            import matplotlib.pyplot as plt

            existing_figs = set(plt.get_fignums())
            original_backend = matplotlib.get_backend()
            with suppress(Exception):
                matplotlib.use("Agg", force=True)
            has_matplotlib = True

            # Hook plt.show() to capture figures BEFORE they're potentially closed
            original_show = plt.show

            def capturing_show(*_args, **_kwargs):
                """Capture all current figures when show() is called."""
                nonlocal captured_figures, captured_fig_ids
                print(f"🔍 [capturing_show] Called! Current figs: {plt.get_fignums()}")
                for fig_num in plt.get_fignums():
                    if fig_num not in existing_figs:
                        fig = plt.figure(fig_num)
                        fig_id = id(fig)
                        print(
                            f"🔍 [capturing_show] Fig {fig_num}: id={fig_id}, already_captured={fig_id in captured_fig_ids}"
                        )
                        # Skip if already captured
                        if fig_id in captured_fig_ids:
                            print("  ⏭️  Skipping (already captured)")
                            continue
                        captured_fig_ids.add(fig_id)
                        buf = io.BytesIO()
                        try:
                            fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
                            buf.seek(0)
                            png_size = len(buf.getvalue())
                            captured_figures.append(
                                CapturedFigure(
                                    {
                                        "figure": None,  # Don't keep reference to avoid issues
                                        "png_bytes": buf.getvalue(),
                                    }
                                )
                            )
                            print(
                                f"  ✅ Captured! PNG size: {png_size} bytes, total captured: {len(captured_figures)}"
                            )
                        except Exception as e:
                            print(f"  ❌ Capture failed: {e}")
                # Don't call original show - we're in Agg backend anyway

            plt.show = capturing_show

        except ImportError:
            has_matplotlib = False
            existing_figs = set()

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                mock_result = self.func(*mock_args, **mock_kwargs)

            # Capture any NEW matplotlib figures that weren't captured by show()
            if has_matplotlib:
                # Restore original show
                plt.show = original_show

                # Collect figures to capture: new figures AND figures from returned Axes
                figures_to_capture = set()

                # Add any new figures
                new_figs = set(plt.get_fignums()) - existing_figs
                for fig_num in new_figs:
                    figures_to_capture.add(plt.figure(fig_num))

                # Also check if result contains Axes - if so, capture their figures
                def extract_axes_figures(obj):
                    """Extract parent figures from Axes objects in result."""
                    figs = set()
                    try:
                        from matplotlib.axes import Axes

                        if isinstance(obj, Axes):
                            if obj.figure and obj.figure.number not in existing_figs:
                                figs.add(obj.figure)
                        elif isinstance(obj, (list, tuple)):
                            for item in obj:
                                figs.update(extract_axes_figures(item))
                        elif isinstance(obj, dict):
                            for item in obj.values():
                                figs.update(extract_axes_figures(item))
                    except Exception:
                        pass
                    return figs

                figures_to_capture.update(extract_axes_figures(mock_result))

                print(f"🔍 [post-execution] Figures to capture: {len(figures_to_capture)}")
                print(f"🔍 [post-execution] Already captured IDs: {captured_fig_ids}")

                # Capture all unique figures (reuse captured_fig_ids for deduplication)
                for fig in figures_to_capture:
                    fig_id = id(fig)
                    print(
                        f"🔍 [post-execution] Fig: id={fig_id}, already_captured={fig_id in captured_fig_ids}"
                    )
                    if fig_id in captured_fig_ids:
                        print("  ⏭️  Skipping (already captured)")
                        continue
                    captured_fig_ids.add(fig_id)
                    buf = io.BytesIO()
                    try:
                        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
                        buf.seek(0)
                        png_size = len(buf.getvalue())
                        captured_figures.append(
                            CapturedFigure(
                                {
                                    "figure": fig,
                                    "png_bytes": buf.getvalue(),
                                }
                            )
                        )
                        print(
                            f"  ✅ Captured! PNG size: {png_size} bytes, total captured: {len(captured_figures)}"
                        )
                    except Exception:
                        pass
                    plt.close(fig)

                # Also close any figures that were captured via show() hook
                plt.close("all")

                with suppress(Exception):
                    matplotlib.use(original_backend, force=True)

            mock_error = None

        except Exception as e:
            mock_result = None
            mock_error = str(e)
            if has_matplotlib:
                # Close any figures that were created before the error
                new_figs = set(plt.get_fignums()) - existing_figs
                for fig_num in new_figs:
                    plt.close(fig_num)
                with suppress(Exception):
                    matplotlib.use(original_backend, force=True)

        # Deduplicate captured figures by content (ignore PNG metadata differences)
        import hashlib

        seen_signatures = set()
        deduped_figures = []

        def figure_signature(png_bytes: bytes):
            """Stable signature of figure pixels to catch duplicates with different metadata."""
            if not png_bytes:
                return None
            try:
                from PIL import Image

                with Image.open(io.BytesIO(png_bytes)) as img:
                    raw_bytes = img.tobytes()
                    digest = hashlib.sha256()
                    digest.update(str(img.size).encode())
                    digest.update(img.mode.encode())
                    digest.update(raw_bytes)
                    return digest.digest()
            except Exception:
                return hashlib.sha256(png_bytes).digest()

        for fig in captured_figures:
            png_bytes = fig.png_bytes if hasattr(fig, "png_bytes") else None
            signature = figure_signature(png_bytes) if png_bytes else None
            if signature:
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
            deduped_figures.append(fig)

        # Capture stdout/stderr after adding debug logs
        public_stdout = stdout_capture.getvalue()
        public_stderr = stderr_capture.getvalue()

        # Wrap in Twin with only public side
        # Handle None results - use a sentinel dict to allow Twin creation
        # Functions that return None (like plotting functions) are valid
        public_value = mock_result
        if public_value is None:
            if mock_error:
                public_value = {"_mock_error": True, "error": mock_error}
            else:
                # Function returned None legitimately (e.g., plotting function)
                # Use a sentinel so Twin can be created
                public_value = {"_none_result": True, "has_figures": len(deduped_figures) > 0}

        result_twin = Twin(
            private=None,
            public=public_value,
            owner=context.user if context else "unknown",
            name=self.result_name,
            twin_id=self.result_id,
        )

        # Attach captured outputs to Twin
        result_twin.public_stdout = public_stdout
        result_twin.public_stderr = public_stderr
        result_twin.public_figures = deduped_figures

        # Return ComputationResult with session reference if we have one
        session = getattr(self, "_session", None)
        if session is None and context is not None:
            session = getattr(context, "_active_session", None)
            if session is not None:
                self._session = session
        return ComputationResult(
            result=result_twin,
            stdout=public_stdout,
            stderr=public_stderr,
            error=mock_error,
            var_name=self.result_name,
            var_id=self.result_id,
            result_id=self.result_id,
            comp_id=self.comp_id,
            sender=self.sender,
            context=context,
            session=session,
        )

    def run_both(self, context=None):
        """
        Execute on both mock and real data.

        Runs mock first, then real. Returns ComputationResult where .data is
        a Twin with both sides for comparison.
        """
        from .remote_vars import _ensure_sparse_shapes, _SafeDisplayProxy
        from .twin import _TWIN_REGISTRY, CapturedFigure, Twin

        # Check for missing imports before execution
        missing = _check_missing_imports_with_versions(self.func, self.required_versions)
        if missing:
            func_name = getattr(self.func, "__name__", "computation")
            if not _prompt_install_function_deps(missing, func_name):
                # User declined - raise helpful error
                import shutil

                use_uv = _is_uv_venv() and shutil.which("uv")
                pip_cmd = "uv pip install" if use_uv else "pip install"
                specs = [f"{p}=={v}" if v else p for p, v in missing]
                raise ImportError(
                    f"\n\n❌ Cannot execute '{func_name}' - missing dependencies\n\n"
                    f"   Required packages not installed:\n"
                    + "\n".join(f"   • {s}" for s in specs)
                    + f"\n\n   To fix, run:\n"
                    f"   {pip_cmd} {' '.join(specs)}\n"
                )

        def unwrap_for_computation(val):
            """Unwrap display proxies and ensure sparse shapes for computation."""
            if isinstance(val, _SafeDisplayProxy):
                val = val._raw
            return _ensure_sparse_shapes(val)

        def get_local_twin_private(arg):
            """Look up local Twin and get its private data."""
            if context:
                twin_id = arg.twin_id
                owner = arg.owner

                # Check global Twin registry for the executing user's version
                key = (twin_id, context.user)
                if key in _TWIN_REGISTRY:
                    local_twin = _TWIN_REGISTRY[key]
                    if local_twin.has_private:
                        return unwrap_for_computation(local_twin.private)

                # Also check with arg's owner
                if owner != context.user:
                    key = (twin_id, owner)
                    if key in _TWIN_REGISTRY:
                        local_twin = _TWIN_REGISTRY[key]
                        if local_twin.has_private:
                            return unwrap_for_computation(local_twin.private)

                # Fallback: any Twin with matching id
                for (tid, _owner), local_twin in _TWIN_REGISTRY.items():
                    if tid == twin_id and local_twin.has_private:
                        return unwrap_for_computation(local_twin.private)

            # Fall back to received Twin's private
            if arg.has_private:
                return unwrap_for_computation(arg.private)
            return None

        # Auto-detect context if needed
        if context is None:
            context = self._auto_detect_context()

        # Run mock first
        print("🧪 Step 1/2: Testing on mock/public data...")
        mock_comp_result = self.run_mock(context=context)

        # Then run on real/private data with clean figure capture
        print("🔒 Step 2/2: Executing on real/private data...")

        def resolve_twin_ref(val):
            """Convert Twin reference dicts back into local Twin instances if available."""
            if isinstance(val, dict) and val.get("_beaver_twin_ref"):
                twin_id = val["twin_id"]
                owner_hint = val.get("owner")

                if owner_hint and (twin_id, owner_hint) in _TWIN_REGISTRY:
                    return _TWIN_REGISTRY[(twin_id, owner_hint)]
                if context and (twin_id, context.user) in _TWIN_REGISTRY:
                    return _TWIN_REGISTRY[(twin_id, context.user)]

                for (tid, _owner), twin in _TWIN_REGISTRY.items():
                    if tid == twin_id:
                        return twin

                if context and hasattr(context, "remote_vars"):
                    for var in context.remote_vars.vars.values():
                        sv = getattr(var, "_stored_value", None)
                        if isinstance(sv, Twin) and sv.twin_id == twin_id:
                            _TWIN_REGISTRY[(sv.twin_id, sv.owner)] = sv
                            return sv

                raise ValueError(
                    f"Twin reference '{val.get('name', 'unknown')}' (ID: {twin_id[:12]}...) not available locally"
                )
            return val

        resolved_args = tuple(resolve_twin_ref(a) for a in self.args)
        resolved_kwargs = {k: resolve_twin_ref(v) for k, v in self.kwargs.items()}

        # Also resolve function references (TwinFunc callbacks)
        from .twin_func import resolve_func_ref

        resolved_args = tuple(resolve_func_ref(a, context) for a in resolved_args)
        resolved_kwargs = {k: resolve_func_ref(v, context) for k, v in resolved_kwargs.items()}

        # Build private args (similar to run_mock but using private data)
        private_args = []
        for arg in resolved_args:
            if isinstance(arg, Twin):
                private_data = get_local_twin_private(arg)
                if private_data is not None:
                    private_args.append(private_data)
                else:
                    raise ValueError("Twin argument has no private data for real execution")
            else:
                private_args.append(arg)

        private_kwargs = {}
        for k, v in resolved_kwargs.items():
            if isinstance(v, Twin):
                private_data = get_local_twin_private(v)
                if private_data is not None:
                    private_kwargs[k] = private_data
                else:
                    raise ValueError(f"Twin kwarg '{k}' has no private data for real execution")
            else:
                private_kwargs[k] = v

        # Capture stdout, stderr, and matplotlib figures for private execution
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        private_figures = []
        private_fig_ids = set()

        def _capture_private_fig(fig):
            """Save a matplotlib figure to PNG once per figure id."""
            nonlocal private_figures, private_fig_ids
            if fig is None:
                return
            if id(fig) in private_fig_ids:
                return
            private_fig_ids.add(id(fig))
            buf = io.BytesIO()
            try:
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
                buf.seek(0)
                private_figures.append(
                    CapturedFigure(
                        {
                            "figure": None,
                            "png_bytes": buf.getvalue(),
                        }
                    )
                )
            except Exception:
                pass

        try:
            import matplotlib
            import matplotlib.pyplot as plt

            # Close ALL existing figures to ensure clean state
            plt.close("all")
            original_backend = matplotlib.get_backend()
            with suppress(Exception):
                matplotlib.use("Agg", force=True)
            has_matplotlib = True

            # Hook plt.show() to capture figures BEFORE they're potentially closed
            original_show = plt.show

            def capturing_show(*_args, **_kwargs):
                """Capture all current figures when show() is called."""
                for fig_num in plt.get_fignums():
                    fig = plt.figure(fig_num)
                    _capture_private_fig(fig)

            plt.show = capturing_show

        except ImportError:
            has_matplotlib = False

        private_result = None
        private_error = None

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                private_result = self.func(*private_args, **private_kwargs)

            # Capture any remaining matplotlib figures and restore show
            if has_matplotlib:
                plt.show = original_show

                # Collect figures to capture
                figures_to_capture = set()

                # Add all figures (we closed all before, so all are new)
                for fig_num in plt.get_fignums():
                    figures_to_capture.add(plt.figure(fig_num))

                # Also check if result contains Axes - if so, capture their figures
                def extract_axes_figures(obj):
                    """Extract parent figures from Axes objects in result."""
                    figs = set()
                    try:
                        from matplotlib.axes import Axes

                        if isinstance(obj, Axes):
                            if obj.figure:
                                figs.add(obj.figure)
                        elif isinstance(obj, (list, tuple)):
                            for item in obj:
                                figs.update(extract_axes_figures(item))
                        elif isinstance(obj, dict):
                            for item in obj.values():
                                figs.update(extract_axes_figures(item))
                    except Exception:
                        pass
                    return figs

                figures_to_capture.update(extract_axes_figures(private_result))

                # Capture all unique figures
                for fig in figures_to_capture:
                    _capture_private_fig(fig)
                    plt.close(fig)

                plt.close("all")
                with suppress(Exception):
                    matplotlib.use(original_backend, force=True)

            private_stdout = stdout_capture.getvalue()
            private_stderr = stderr_capture.getvalue()

        except Exception as e:
            private_result = None
            private_error = str(e)
            private_stdout = stdout_capture.getvalue()
            private_stderr = stderr_capture.getvalue() + f"\n{e}"
            if has_matplotlib:
                plt.show = original_show
                plt.close("all")
                with suppress(Exception):
                    matplotlib.use(original_backend, force=True)

        print(f"✓ Private result: {type(private_result).__name__}")

        # Extract the mock data from the Twin returned by run_mock()
        mock_twin = mock_comp_result.result
        mock_data = mock_twin.public if isinstance(mock_twin, Twin) else mock_twin

        # Handle None results - use sentinel dicts to allow Twin creation
        # Functions that return None (like plotting functions) are valid
        public_value = mock_data
        private_value = private_result

        if public_value is None:
            public_value = {
                "_none_result": True,
                "has_figures": len(getattr(mock_twin, "public_figures", []) or []) > 0,
            }
        if private_value is None:
            if private_error:
                private_value = {"_private_error": True, "error": private_error}
            else:
                private_value = {"_none_result": True, "has_figures": len(private_figures) > 0}

        # Create Twin with both sides
        result_twin = Twin(
            public=public_value,
            private=private_value,
            owner=context.user if context else "unknown",
            name=self.result_name,
            twin_id=self.result_id,
        )

        # Attach captured outputs to Twin
        # Public outputs from mock run
        try:
            result_twin.public_stdout = object.__getattribute__(mock_twin, "public_stdout")
        except AttributeError:
            result_twin.public_stdout = mock_comp_result.stdout
        try:
            result_twin.public_stderr = object.__getattribute__(mock_twin, "public_stderr")
        except AttributeError:
            result_twin.public_stderr = mock_comp_result.stderr
        try:
            result_twin.public_figures = object.__getattribute__(mock_twin, "public_figures")
        except AttributeError:
            result_twin.public_figures = []

        # Private outputs from real run
        result_twin.private_stdout = private_stdout
        result_twin.private_stderr = private_stderr
        result_twin.private_figures = private_figures

        # Create ComputationResult with session reference if we have one
        session = getattr(self, "_session", None)
        if session is None and context is not None:
            # Fallback to the active session from the executing context if available
            session = getattr(context, "_active_session", None)
            if session is not None:
                self._session = session
        return ComputationResult(
            result=result_twin,
            stdout=private_stdout,
            stderr=private_stderr,
            error=private_error,
            var_name=self.result_name,
            var_id=self.result_id,
            comp_id=self.comp_id,
            sender=self.sender,
            context=context,
            session=session,
        )

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

        # Check for global state access
        global_refs = _detect_global_access(self.func)
        if global_refs:
            lines.append("")
            lines.append("\033[33m⚠️  WARNING: Function accesses global state!\033[0m")
            lines.append("   These variables are NOT passed as parameters:")
            for ref in global_refs[:5]:  # Show first 5
                lines.append(f"   • \033[33m{ref}\033[0m")
            if len(global_refs) > 5:
                lines.append(f"   ... and {len(global_refs) - 5} more")
            lines.append(
                "   \033[33m💡 This may cause different results on remote execution!\033[0m"
            )

        # Bound Data section - show what data is bound to this action
        data_lines = _describe_bound_data(
            self.args, self.kwargs, context=self._auto_detect_context()
        )

        if data_lines:
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
                        # Simple inline storage (scalars, etc.)
                        if var._stored_value is not None:
                            return var._stored_value

                        # Twin: prefer owner's held instance, otherwise load published copy
                        if str(var.var_type).startswith("Twin"):
                            if hasattr(var, "_twin_reference"):
                                return var._twin_reference
                            if var.data_location:
                                from .runtime import read_envelope

                                env = read_envelope(var.data_location)
                                return env.load(inject=False)

                        raise ValueError(
                            f"Remote var '{var.name}' (ID: {var_id[:12]}...) has no stored value"
                        )
                raise ValueError(f"Remote var ID {var_id[:12]}... not found in registry")
            else:
                raise ValueError("Cannot resolve remote var reference without context")

        # Check if this is a Twin reference dict (from ComputationRequest serialization)
        if isinstance(arg, dict) and arg.get("_beaver_twin_ref"):
            from .twin import _TWIN_REGISTRY

            if context:
                twin_id = arg["twin_id"]
                owner = arg.get("owner", context.user)

                # Try to find the owner's version of this Twin
                key = (twin_id, context.user)
                if key in _TWIN_REGISTRY:
                    registered_twin = _TWIN_REGISTRY[key]
                    if registered_twin.has_private:
                        return registered_twin.private

                # Also check with arg's owner
                if owner != context.user:
                    key = (twin_id, owner)
                    if key in _TWIN_REGISTRY:
                        registered_twin = _TWIN_REGISTRY[key]
                        if registered_twin.has_private:
                            return registered_twin.private

                raise ValueError(
                    f"Twin reference '{arg.get('name', 'unknown')}' "
                    f"(ID: {twin_id[:12]}...) not found in registry"
                )
            else:
                raise ValueError("Cannot resolve Twin reference without context")

        # Check if this is a Twin - look it up in global Twin registry and unwrap to private data
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
                        # Unwrap to private data - function expects raw data, not Twin
                        return registered_twin.private

                # Also check with arg's owner (in case it differs from context.user)
                if owner != context.user:
                    key = (twin_id, owner)
                    if key in _TWIN_REGISTRY:
                        registered_twin = _TWIN_REGISTRY[key]
                        if registered_twin.has_private:
                            # Unwrap to private data
                            return registered_twin.private

            # Use the received Twin's private data if available
            if arg.has_private:
                return arg.private
            # Fall back to public if no private (e.g., for mock scenarios)
            if arg.has_public:
                from .remote_vars import _ensure_sparse_shapes, _SafeDisplayProxy

                val = arg.public
                # Unwrap SafeDisplayProxy to get raw object
                if isinstance(val, _SafeDisplayProxy):
                    val = val._raw
                return _ensure_sparse_shapes(val)
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
            self.staged.remove(computation)

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
