"""Lightweight notebook-to-notebook ferry for Python objects using Fory."""

# Debug flag for beaver output
# Also controls SyftBox SDK debug output when using SyftBoxBackend
debug = False

# Enable matplotlib inline mode by default in Jupyter/IPython
try:
    from IPython import get_ipython

    ip = get_ipython()
    if ip is not None:
        ip.run_line_magic("matplotlib", "inline")
except (ImportError, AttributeError):
    pass

from .computation import (
    ComputationRequest,
    ComputationResult,
    RemoteComputationPointer,
    StagingArea,
    execute_remote_computation,
)
from .envelope import BeaverEnvelope
from .policy import (
    PERMISSIVE_POLICY,
    STRICT_POLICY,
    VERBOSE_POLICY,
    BeaverPolicy,
)
from .remote_vars import RemoteVar, RemoteVarRegistry, RemoteVarView
from .session import Session, SessionRequest, SessionRequestsView
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
from .twin import CapturedFigure, Twin
from .twin_result import TwinComputationResult

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
    "import_peer_bundle",
    "load_by_id",
    "listen_once",
    "pack",
    "provision_identity",
    "read_envelope",
    "unpack",
    "wait_for_reply",
    "write_envelope",
]
