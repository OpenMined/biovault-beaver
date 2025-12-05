"""Lightweight notebook-to-notebook ferry for Python objects using Fory."""

from . import sample_data
from .computation import (
    ComputationRequest,
    ComputationResult,
    RemoteComputationPointer,
    StagingArea,
    execute_remote_computation,
)
from .envelope import BeaverEnvelope
from .lib_support import register_builtin_loader
from .policy import (
    PERMISSIVE_POLICY,
    STRICT_POLICY,
    VERBOSE_POLICY,
    BeaverPolicy,
)
from .remote_vars import RemoteVar, RemoteVarRegistry, RemoteVarView
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
from .twin_result import TwinComputationResult

# Debug flag for beaver output
# Also controls SyftBox SDK debug output when using SyftBoxBackend
_debug_enabled = False


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


# Enable matplotlib inline mode by default in Jupyter/IPython
try:
    from IPython import get_ipython

    ip = get_ipython()
    if ip is not None:
        ip.run_line_magic("matplotlib", "inline")
except (ImportError, AttributeError):
    pass

# SyftBox integration (optional - requires syftbox-sdk)
try:
    from .syftbox_backend import (
        SyftBoxBackend,
        import_peer_bundle,
        provision_identity,
    )

    _SYFTBOX_AVAILABLE = True
except ImportError:
    _SYFTBOX_AVAILABLE = False
    SyftBoxBackend = None  # type: ignore
    import_peer_bundle = None  # type: ignore
    provision_identity = None  # type: ignore

__version__ = "0.1.23"
__all__ = [
    "BeaverEnvelope",
    "BeaverContext",
    "BeaverPolicy",
    "CapturedFigure",
    "ComputationRequest",
    "ComputationResult",
    "PERMISSIVE_POLICY",
    "VERBOSE_POLICY",
    "STRICT_POLICY",
    "RemoteComputationPointer",
    "RemoteVar",
    "RemoteVarRegistry",
    "RemoteVarView",
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
    "execute_remote_computation",
    "export",
    "find_by_id",
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
