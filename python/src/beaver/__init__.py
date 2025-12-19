"""Lightweight notebook-to-notebook ferry for Python objects using Fory."""

from .computation import (
    ComputationRequest,
    ComputationResult,
    RemoteComputationPointer,
    StagingArea,
    execute_remote_computation,
)
from .datasets import DatasetRegistry
from .envelope import BeaverEnvelope
from .lib_support import register_builtin_loader
from .live_var import LiveVar
from .mappings import MappingStore
from .policy import (
    PERMISSIVE_POLICY,
    STRICT_POLICY,
    VERBOSE_POLICY,
    BeaverPolicy,
)
from .remote_data import get_current_owner
from .remote_vars import NotReady, RemoteVar, RemoteVarRegistry, RemoteVarView
from .runtime import (
    BeaverContext,
    InboxView,
    SendResult,
    TrustedLoader,
    connect,
    export,
    find_by_id,
    list_inbox,
    listen_once,
    load,
    load_by_id,
    pack,
    read_envelope,
    save,
    snapshot,
    unpack,
    wait_for_reply,
    write_envelope,
)
from .session import Session, SessionRequest, SessionRequestsView
from .twin import CapturedFigure, Twin
from .twin_func import TwinFunc, func_to_ref, resolve_func_ref
from .twin_result import TwinComputationResult

# Debug flag for beaver output
# Also controls SyftBox SDK debug output when using SyftBoxBackend
_debug_enabled = False


def __getattr__(name: str):
    """
    Lazy imports for optional/heavier modules.

    This keeps `import beaver` lightweight so basic serialization can work in minimal
    environments and tests can control optional deps.
    """
    if name == "sample_data":
        from . import sample_data as _sample_data  # noqa: PLC0415

        return _sample_data
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def set_debug(enabled: bool = True) -> None:
    """Enable or disable debug mode."""
    global _debug_enabled
    _debug_enabled = enabled


def get_debug() -> bool:
    """Get current debug state."""
    return _debug_enabled


# For backwards compatibility
debug = _debug_enabled


def show_debug() -> None:
    """Print debug information about beaver environment and configuration."""
    import os
    import sys
    from pathlib import Path

    print("=" * 60)
    print("🦫 BEAVER DEBUG INFO")
    print("=" * 60)

    print("\n📦 Package Info:")
    print(f"  Version: {__version__}")
    print(f"  Debug enabled: {_debug_enabled}")
    print(f"  SyftBox available: {_SYFTBOX_AVAILABLE}")

    print("\n🌍 Environment Variables:")
    env_vars = [
        "SYFTBOX_EMAIL",
        "SYFTBOX_DATA_DIR",
        "BEAVER_SESSION_ID",
        "BIOVAULT_HOME",
        "SYFTBOX_CONFIG_PATH",
        "SYFTBOX_SERVER_URL",
        "VIRTUAL_ENV",
        "PWD",
    ]
    for var in env_vars:
        value = os.environ.get(var, "<not set>")
        print(f"  {var}: {value}")

    print("\n📁 Working Directory:")
    print(f"  {os.getcwd()}")

    print("\n🐍 Python:")
    print(f"  Version: {sys.version}")
    print(f"  Executable: {sys.executable}")

    # Check for session.json in cwd
    session_json = Path(os.getcwd()) / "session.json"
    if session_json.exists():
        print("\n📄 Session Config (session.json):")
        try:
            import json

            with open(session_json) as f:
                config = json.load(f)
            for k, v in config.items():
                print(f"  {k}: {v}")
        except Exception as e:
            print(f"  Error reading: {e}")
    else:
        print("\n📄 Session Config: Not found (no session.json in cwd)")

    print("\n" + "=" * 60)


def dev() -> None:
    """Reinstall beaver and syftbox-sdk from local dev paths for development.

    Uses BIOVAULT_HOME environment variable to find the packages.
    Requires kernel restart after running to use new versions.
    """
    import os
    import subprocess

    biovault_home = os.environ.get("BIOVAULT_HOME", "")
    if not biovault_home:
        print("BIOVAULT_HOME not set. Cannot find dev packages.")
        return

    base_repo = os.path.normpath(os.path.join(biovault_home, "..", ".."))

    # Reinstall syftbox-sdk
    syftbox_path = os.path.join(base_repo, "syftbox-sdk", "python")
    if os.path.exists(syftbox_path):
        subprocess.run(["uv", "pip", "install", "-e", syftbox_path, "-q"], check=True)
        print(f"syftbox-sdk reinstalled from {syftbox_path}")
    else:
        print(f"syftbox-sdk not found at {syftbox_path}")

    # Reinstall beaver
    beaver_path = os.path.join(base_repo, "biovault-beaver", "python")
    if os.path.exists(beaver_path):
        subprocess.run(["uv", "pip", "install", "-e", beaver_path, "-q"], check=True)
        print(f"beaver reinstalled from {beaver_path}")
    else:
        print(f"beaver not found at {beaver_path}")

    print("\nRestart kernel to use new versions.")


# Enable matplotlib inline mode by default in Jupyter/IPython.
# Avoid importing IPython in non-interactive scripts (it can trigger heavy imports).
try:
    import sys

    if "IPython" in sys.modules:
        from IPython import get_ipython  # type: ignore

        ip = get_ipython()
        if ip is not None:
            ip.run_line_magic("matplotlib", "inline")
except Exception:
    pass

# SyftBox integration (optional - requires syftbox-sdk)
try:
    from .syftbox_backend import (
        SenderVerificationError,
        SyftBoxBackend,
        import_peer_bundle,
        provision_identity,
    )

    _SYFTBOX_AVAILABLE = True
except ImportError:
    _SYFTBOX_AVAILABLE = False
    SenderVerificationError = None  # type: ignore
    SyftBoxBackend = None  # type: ignore
    import_peer_bundle = None  # type: ignore
    provision_identity = None  # type: ignore


def active_session():
    """
    Get the currently active session, if any.

    This is a convenience function that creates a BeaverContext and
    checks for an active session from:
    1. BEAVER_SESSION_ID environment variable
    2. session.json file in current working directory

    Returns:
        Session object if an active session exists, None otherwise

    Example:
        import beaver
        session = beaver.active_session()
        if session:
            session.send(my_data)
    """
    import os
    from pathlib import Path

    def _auto_load_state(sess):
        # Optional auto-restore of private state for convenience (opt-in)
        flag = os.environ.get("BEAVER_AUTO_LOAD_STATE", "0").lower()
        if flag in ("1", "true", "yes", "on"):
            try:
                sess.load()
            except FileNotFoundError:
                pass
            except Exception as e:  # noqa: BLE001
                if debug:
                    print(f"[DEBUG] auto load_state failed: {e}")
        return sess

    # Check if there's a session to load
    session_id = os.environ.get("BEAVER_SESSION_ID")
    session_json_path = Path(os.getcwd()) / "session.json"

    if not session_id and not session_json_path.exists():
        return None

    # Create context (will auto-detect data_dir from env)
    try:
        ctx = connect()
        sess = ctx.active_session()
        if sess:
            return _auto_load_state(sess)
        return None
    except Exception as e:
        print(f"⚠️  Failed to load active session: {e}")
        return None


# Cache for the auto-detected context
_auto_context = None


def _get_auto_context():
    """Get or create an auto-detected BeaverContext."""
    global _auto_context
    if _auto_context is None:
        _auto_context = connect()
    return _auto_context


def ctx():
    """
    Get the auto-detected BeaverContext.

    Returns a cached context that auto-detects configuration from
    environment variables (SYFTBOX_EMAIL, SYFTBOX_DATA_DIR, etc).

    Example:
        import beaver
        bv = beaver.ctx()

        @bv
        def analyze(data):
            return data.mean()
    """
    return _get_auto_context()


class _FnDecorator:
    """
    Smart decorator that auto-detects context.

    Usage:
        import beaver

        @beaver.fn
        def analyze(data):
            return data.mean()

        # Or with options:
        @beaver.fn(name="my_analysis")
        def analyze(data):
            return data.mean()
    """

    def __call__(self, func=None, **kwargs):
        """Use as @beaver.fn or @beaver.fn(options)."""
        context = _get_auto_context()
        return context.__call__(func, **kwargs)

    def __repr__(self):
        return "<beaver.fn decorator - use @beaver.fn to decorate functions>"


# Module-level decorator instance
fn = _FnDecorator()


__version__ = "0.1.43"
__all__ = [
    "sample_data",
    "active_session",
    "BeaverContext",
    "BeaverEnvelope",
    "BeaverPolicy",
    "ctx",
    "fn",
    "CapturedFigure",
    "ComputationRequest",
    "ComputationResult",
    "DatasetRegistry",
    "MappingStore",
    "PERMISSIVE_POLICY",
    "VERBOSE_POLICY",
    "STRICT_POLICY",
    "RemoteComputationPointer",
    "RemoteVar",
    "RemoteVarRegistry",
    "RemoteVarView",
    "SenderVerificationError",
    "Session",
    "SessionRequest",
    "SessionRequestsView",
    "register_builtin_loader",
    "StagingArea",
    "SendResult",
    "SyftBoxBackend",
    "TrustedLoader",
    "Twin",
    "TwinComputationResult",
    "connect",
    "dev",
    "execute_remote_computation",
    "export",
    "find_by_id",
    "get_current_owner",
    "get_debug",
    "import_peer_bundle",
    "load_by_id",
    "listen_once",
    "pack",
    "provision_identity",
    "read_envelope",
    "set_debug",
    "show_debug",
    "unpack",
    "wait_for_reply",
    "write_envelope",
    "sample_data",
]
