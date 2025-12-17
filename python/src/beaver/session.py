"""Session management for Beaver + SyftBox integration.

Sessions provide a structured way for data scientists and data owners to
establish encrypted communication channels with proper permissions.

Session lifecycle:
1. DS requests session with DO via RPC
2. DO reviews and accepts session
3. Mirrored session folders are created on both sides
4. Each party writes to their own folder, reads from peer's folder
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from uuid import uuid4

if TYPE_CHECKING:
    from .datasets import Dataset
    from .runtime import BeaverContext
    from .twin import Twin


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generate_session_id() -> str:
    """Generate a short random session ID."""
    return hashlib.sha256(uuid4().bytes).hexdigest()[:12]


def _pkg_version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except Exception:
        return None


def _version_info() -> dict[str, Optional[str]]:
    info: dict[str, Optional[str]] = {}
    info["beaver"] = _pkg_version("biovault-beaver") or _pkg_version("beaver")
    info["syftbox_sdk"] = _pkg_version("syftbox-sdk")
    info["python"] = sys.version.split()[0]
    info["platform"] = platform.platform()
    return info


class SessionWorkspace:
    """View of a session's workspace showing both local and peer data."""

    def __init__(self, session, local_files, peer_files):
        self.session = session
        self.local_files = local_files
        self.peer_files = peer_files

    def __repr__(self) -> str:
        lines = [f"SessionWorkspace: {self.session.session_id}"]
        lines.append(f"  Peer: {self.session.peer}")
        lines.append("")

        lines.append(f"📤 Our data ({len(self.local_files)} files):")
        if self.local_files:
            for f in self.local_files:
                lines.append(f"   - {f.name or f.envelope_id[:12]}")
        else:
            lines.append("   (empty)")

        lines.append("")
        lines.append(f"📥 Peer's data ({len(self.peer_files)} files):")
        if self.peer_files:
            for f in self.peer_files:
                lines.append(f"   - {f.name or f.envelope_id[:12]}")
        else:
            lines.append("   (empty)")

        return "\n".join(lines)

    def _repr_html_(self) -> str:
        html = [f"<h4>SessionWorkspace: {self.session.session_id}</h4>"]
        html.append(f"<p><b>Peer:</b> {self.session.peer}</p>")

        html.append(f"<h5>📤 Our data ({len(self.local_files)} files)</h5>")
        if self.local_files:
            html.append("<ul>")
            for f in self.local_files:
                html.append(f"<li>{f.name or f.envelope_id[:12]}</li>")
            html.append("</ul>")
        else:
            html.append("<p><i>(empty)</i></p>")

        html.append(f"<h5>📥 Peer's data ({len(self.peer_files)} files)</h5>")
        if self.peer_files:
            html.append("<ul>")
            for f in self.peer_files:
                html.append(f"<li>{f.name or f.envelope_id[:12]}</li>")
            html.append("</ul>")
        else:
            html.append("<p><i>(empty)</i></p>")

        return "".join(html)


@dataclass
class SessionRequest:
    """
    A pending session request from a data scientist.

    Data owners receive these requests and can accept/reject them.
    """

    session_id: str
    requester: str  # Email of the requester (DS)
    target: str  # Email of the target (DO)
    created_at: str = field(default_factory=_iso_now)
    message: Optional[str] = None  # Optional message from requester
    status: str = "pending"  # pending, accepted, rejected

    # Internal reference to context for .accept() method
    _context: Optional[BeaverContext] = field(default=None, repr=False)

    def accept(self) -> Session:
        """
        Accept this session request.

        Creates the session folders and sends acceptance response.

        Returns:
            Session object for the accepted session
        """
        if self._context is None:
            raise RuntimeError(
                "SessionRequest not bound to a context. Use bv.accept_session() instead."
            )

        return self._context.accept_session(self)

    def reject(self, reason: Optional[str] = None) -> None:
        """
        Reject this session request.

        Args:
            reason: Optional reason for rejection
        """
        if self._context is None:
            raise RuntimeError("SessionRequest not bound to a context.")

        self._context._reject_session(self, reason=reason)

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "session_id": self.session_id,
            "requester": self.requester,
            "target": self.target,
            "created_at": self.created_at,
            "message": self.message,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict, context: Optional[BeaverContext] = None) -> SessionRequest:
        """Create from dict."""
        req = cls(
            session_id=data["session_id"],
            requester=data["requester"],
            target=data["target"],
            created_at=data.get("created_at", _iso_now()),
            message=data.get("message"),
            status=data.get("status", "pending"),
        )
        req._context = context
        return req

    def __repr__(self) -> str:
        status_icon = {"pending": "⏳", "accepted": "✅", "rejected": "❌"}.get(self.status, "❓")
        return (
            f"{status_icon} SessionRequest(\n"
            f"    session_id={self.session_id!r},\n"
            f"    from={self.requester!r},\n"
            f"    status={self.status!r},\n"
            f"    created_at={self.created_at[:19]!r}\n"
            f")"
        )

    def _repr_html_(self) -> str:
        status_icon = {"pending": "⏳", "accepted": "✅", "rejected": "❌"}.get(self.status, "❓")
        msg_html = f"<br><i>Message: {self.message}</i>" if self.message else ""
        return (
            f"<div style='border:1px solid #ccc; padding:10px; margin:5px; border-radius:5px;'>"
            f"<b>{status_icon} Session Request</b><br>"
            f"<code>ID: {self.session_id}</code><br>"
            f"From: <b>{self.requester}</b><br>"
            f"Status: {self.status}<br>"
            f"Created: {self.created_at[:19]}"
            f"{msg_html}"
            f"</div>"
        )


@dataclass
class SessionDatasetInfo:
    """Information about a dataset associated with a session."""

    owner: str
    name: str
    public_url: str
    role: str = "shared"  # 'shared' (using shared data) or 'yours' (your own data)

    def to_dict(self) -> dict:
        return {
            "owner": self.owner,
            "name": self.name,
            "public_url": self.public_url,
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SessionDatasetInfo:
        return cls(
            owner=data.get("owner", ""),
            name=data.get("name", ""),
            public_url=data.get("public_url", ""),
            role=data.get("role", "shared"),
        )


@dataclass
class Session:
    """
    An active session between a data scientist and data owner.

    Provides access to the shared session folder for data exchange.
    """

    session_id: str
    peer: str  # The other party in the session
    owner: str  # Our email
    role: str  # "requester" (DS) or "accepter" (DO)
    created_at: str = field(default_factory=_iso_now)
    accepted_at: Optional[str] = None
    status: str = "pending"  # pending, active, closed
    datasets: list[SessionDatasetInfo] = field(default_factory=list)  # Associated datasets

    # Internal references
    _context: Optional[BeaverContext] = field(default=None, repr=False)
    _local_path: Optional[Path] = field(default=None, repr=False)
    _peer_path: Optional[Path] = field(default=None, repr=False)
    _datasets_cache: dict[str, Dataset] = field(default_factory=dict, repr=False)

    @property
    def context(self) -> Optional[BeaverContext]:
        """
        Get the BeaverContext for this session.

        Use this to access beaver functionality like the @bv decorator:
            session = beaver.active_session()
            bv = session.context

            @bv
            def my_computation(data):
                return data.mean()
        """
        return self._context

    def ctx(self) -> Optional[BeaverContext]:
        """
        Get the BeaverContext for this session.

        Shorthand matching beaver.ctx() pattern:
            session = beaver.active_session()
            bv = session.ctx()

            @bv
            def my_computation(data):
                return data.mean()
        """
        return self._context

    @property
    def local_folder(self) -> Optional[Path]:
        """Path to our local session folder (we write here)."""
        return self._local_path

    @property
    def peer_folder(self) -> Optional[Path]:
        """Path to peer's session folder (we read from here)."""
        return self._peer_path

    @property
    def is_active(self) -> bool:
        """Check if session is active."""
        return self.status == "active"

    def load_dataset(self, owner: str, name: str) -> Dataset:
        """
        Load a dataset associated with this session.

        Args:
            owner: Dataset owner email
            name: Dataset name

        Returns:
            Dataset object for the specified dataset

        Example:
            session = beaver.active_session()
            dataset = session.load_dataset("owner@example.com", "single_cell")
            twin = dataset["sc_rnaseq"]  # Get a specific asset as Twin
        """
        if self._context is None:
            raise RuntimeError("Session not bound to a context.")

        cache_key = f"{owner}/{name}"
        if cache_key in self._datasets_cache:
            return self._datasets_cache[cache_key]

        # Access via context's dataset registry
        dataset = self._context.datasets[owner][name]
        self._datasets_cache[cache_key] = dataset
        return dataset

    def get_datasets(self) -> dict[str, Dataset]:
        """
        Get all datasets associated with this session.

        Returns:
            Dict mapping "owner/name" to Dataset objects

        Example:
            session = beaver.active_session()
            for key, dataset in session.get_datasets().items():
                print(f"Dataset: {key}")
                for asset in dataset.assets():
                    print(f"  - {asset}")
        """
        if self._context is None:
            raise RuntimeError("Session not bound to a context.")

        result = {}
        for ds_info in self.datasets:
            key = f"{ds_info.owner}/{ds_info.name}"
            try:
                dataset = self.load_dataset(ds_info.owner, ds_info.name)
                result[key] = dataset
            except Exception as e:
                print(f"⚠️  Failed to load dataset {key}: {e}")

        return result

    def get_twins(self) -> dict[str, Twin]:
        """
        Get all assets from all associated datasets as Twins.

        Returns:
            Dict mapping "dataset_name.asset_key" to Twin objects

        Example:
            session = beaver.active_session()
            twins = session.get_twins()
            patient_data = twins["single_cell.sc_rnaseq"]
        """
        if self._context is None:
            raise RuntimeError("Session not bound to a context.")

        result = {}
        for ds_info in self.datasets:
            try:
                dataset = self.load_dataset(ds_info.owner, ds_info.name)
                for asset_key in dataset.assets():
                    twin = dataset[asset_key]
                    result[f"{ds_info.name}.{asset_key}"] = twin
            except Exception as e:
                print(f"⚠️  Failed to load dataset {ds_info.owner}/{ds_info.name}: {e}")

        return result

    def has_datasets(self) -> bool:
        """Check if this session has any associated datasets."""
        return len(self.datasets) > 0

    def open(self) -> None:
        """Open the local session folder in Finder on macOS."""
        if sys.platform != "darwin":
            raise RuntimeError("Session.open() is only supported on macOS.")

        if self._local_path is None:
            self._setup_paths()

        if self._local_path is None or not self._local_path.exists():
            raise FileNotFoundError("Session folder is not available to open.")

        # Use macOS `open` to reveal the folder in Finder
        subprocess.run(["open", str(self._local_path)], check=True)

    def reset(self, *, force: bool = False) -> int:
        """
        Reset the session by deleting all files except syft.pub.yaml.

        Useful for testing when you want to start fresh without recreating
        the session.

        Args:
            force: Must be True to actually delete files (safety check)

        Returns:
            Number of files deleted

        Example:
            session.reset(force=True)  # Delete all beaver files
        """
        if not force:
            print("⚠️  session.reset() requires force=True to delete files")
            print("   This will delete all .beaver files, remote_vars.json, and data/ folder")
            print("   The syft.pub.yaml will be preserved")
            return 0

        if self._local_path is None:
            self._setup_paths()

        if self._local_path is None or not self._local_path.exists():
            print("Session folder doesn't exist")
            return 0

        import shutil

        deleted = 0
        preserve = {"syft.pub.yaml"}

        # Delete files in session folder (local outbox)
        for item in self._local_path.iterdir():
            if item.name in preserve:
                continue

            try:
                if item.is_file():
                    item.unlink()
                    deleted += 1
                    print(f"  Deleted: {item.name}")
                elif item.is_dir():
                    count = sum(1 for _ in item.rglob("*") if _.is_file())
                    shutil.rmtree(item)
                    deleted += count
                    print(f"  Deleted: {item.name}/ ({count} files)")
            except Exception as e:
                print(f"  Failed to delete {item.name}: {e}")

        # Also delete files in inbox (unencrypted peer folder) - incoming requests
        # The inbox is in unencrypted/<peer>/shared/biovault/sessions/<id>/
        if self._context and hasattr(self._context, "_backend") and self._context._backend:
            backend = self._context._backend
            inbox_path = (
                backend.data_dir
                / "unencrypted"
                / self.peer
                / "shared"
                / "biovault"
                / "sessions"
                / self.session_id
            )
            if inbox_path.exists():
                inbox_deleted = 0
                for item in inbox_path.iterdir():
                    if item.name in preserve:
                        continue
                    try:
                        if item.is_file():
                            item.unlink()
                            inbox_deleted += 1
                        elif item.is_dir():
                            count = sum(1 for _ in item.rglob("*") if _.is_file())
                            shutil.rmtree(item)
                            inbox_deleted += count
                    except Exception:
                        pass
                if inbox_deleted > 0:
                    print(f"  Deleted: inbox/ ({inbox_deleted} files)")
                    deleted += inbox_deleted

        # Clear caches
        self._datasets_cache.clear()
        if hasattr(self, "_remote_vars") and self._remote_vars:
            self._remote_vars._vars.clear()
        if hasattr(self, "_peer_remote_vars") and self._peer_remote_vars:
            self._peer_remote_vars._vars.clear()

        print(f"✓ Session reset: {deleted} files deleted")
        return deleted

    def wait_for_acceptance(
        self,
        timeout: float = 60.0,
        poll_interval: float = 2.0,
    ) -> bool:
        """
        Wait for the session to be accepted by the peer.

        Args:
            timeout: Maximum seconds to wait
            poll_interval: Seconds between checks

        Returns:
            True if accepted, False if timeout or rejected
        """
        if self.status == "active":
            return True

        if self._context is None:
            raise RuntimeError("Session not bound to a context.")

        deadline = time.monotonic() + timeout
        print(f"⏳ Waiting for {self.peer} to accept session {self.session_id}...")

        while time.monotonic() < deadline:
            # Check for acceptance response
            if self._check_acceptance():
                print(f"✅ Session {self.session_id} accepted!")
                return True

            time.sleep(poll_interval)

        print("⏰ Timeout waiting for session acceptance")
        return False

    def _check_acceptance(self) -> bool:
        """Check if session has been accepted."""
        if self._context is None or self._context._backend is None:
            return False

        backend = self._context._backend

        # Check for response in RPC folder
        rpc_path = (
            backend.data_dir
            / "datasites"
            / self.owner
            / "app_data"
            / "biovault"
            / "rpc"
            / "session"
        )

        response_file = rpc_path / f"{self.session_id}.response"
        if response_file.exists():
            try:
                # Read and decrypt the response
                if backend.uses_crypto:
                    raw = bytes(backend.storage.read_with_shadow(str(response_file)))
                    data = json.loads(raw.decode("utf-8"))
                else:
                    data = json.loads(response_file.read_text())

                if data.get("status") == "accepted":
                    self.status = "active"
                    self.accepted_at = data.get("accepted_at")
                    self._setup_paths()
                    self._create_local_folder()
                    return True
                elif data.get("status") == "rejected":
                    self.status = "rejected"
                    return False
            except Exception:
                # Could be decryption error, file not fully synced, etc.
                pass

        return False

    def _setup_paths(self) -> None:
        """Setup local and peer folder paths."""
        if self._context is None:
            return

        if self._context._backend is not None:
            backend = self._context._backend

            # Our folder: datasites/<our_email>/shared/biovault/sessions/<session_id>/
            self._local_path = (
                backend.data_dir
                / "datasites"
                / self.owner
                / "shared"
                / "biovault"
                / "sessions"
                / self.session_id
            )

            # Peer's folder: datasites/<peer_email>/shared/biovault/sessions/<session_id>/
            self._peer_path = (
                backend.data_dir
                / "datasites"
                / self.peer
                / "shared"
                / "biovault"
                / "sessions"
                / self.session_id
            )
        else:
            # Local filesystem mode - use inbox_path as base
            base_path = self._context.inbox_path / "sessions" / self.session_id
            self._local_path = base_path
            self._peer_path = base_path  # Same folder in local mode

        # Create our local folder
        self._local_path.mkdir(parents=True, exist_ok=True)
        # Ensure peer session folder exists locally so downloads have a target
        self._peer_path.mkdir(parents=True, exist_ok=True)
        # Ensure data subdirectories exist on both sides to avoid sync rename errors
        (self._local_path / "data").mkdir(parents=True, exist_ok=True)
        (self._peer_path / "data").mkdir(parents=True, exist_ok=True)

        # Create permission file (syft.pub.yaml) - only for backend mode
        if self._context._backend is not None:
            self._write_permission_file()

    def _write_permission_file(self) -> None:
        """Write syft.pub.yaml permission file for the session folder."""
        if self._local_path is None:
            return

        # Permission: peer can READ, owner can WRITE
        # Using proper rules: format for SyftBox
        permission_content = f"""# Session permission file
# Session ID: {self.session_id}
# Created: {self.created_at}

rules:
  - pattern: '**/*'
    access:
      admin:
        - '{self.owner}'
      read:
        - '{self.peer}'
      write:
        - '{self.owner}'
"""
        perm_file = self._local_path / "syft.pub.yaml"

        # Use SyftBox storage if available (writes plaintext for permission files)
        if self._context and hasattr(self._context, "_backend") and self._context._backend:
            backend = self._context._backend
            # Permission files are always plaintext
            backend.storage.write_text(str(perm_file), permission_content, True)
        else:
            perm_file.write_text(permission_content)

    def _create_local_folder(self) -> None:
        """Create the local session folder with proper structure."""
        if self._local_path is None:
            self._setup_paths()

        if self._local_path is None:
            return

        # Create session folder
        self._local_path.mkdir(parents=True, exist_ok=True)

        # Create data subfolder
        (self._local_path / "data").mkdir(exist_ok=True)

        # Write permission file
        self._write_permission_file()

        print(f"📁 Created session folder: {self._local_path}")

    def _default_state_path(self) -> Path:
        """Return a private, non-shared path for session state snapshots."""
        base = os.environ.get("BEAVER_STATE_DIR")
        if base:
            root = Path(base)
        elif os.environ.get("VIRTUAL_ENV"):
            root = Path(os.environ["VIRTUAL_ENV"])
        else:
            root = Path(sys.prefix)
        root = root / "beaver_state"
        root.mkdir(parents=True, exist_ok=True)
        owner_slug = self.owner.replace("@", "_at_").replace("/", "_")
        return root / f"{self.session_id}_{owner_slug}.beaver"

    def save(
        self,
        path: Optional[Path | str] = None,
        *,
        globals_ns: Optional[dict] = None,
        include_private: bool = False,
        strict: bool = False,
        policy=None,
    ) -> Path:
        """
        Snapshot current globals into a private .beaver file for later restore.
        """
        from . import runtime

        target = Path(path) if path is not None else self._default_state_path()
        target.parent.mkdir(parents=True, exist_ok=True)

        ns = globals_ns
        if ns is None:
            try:
                import __main__

                ns = __main__.__dict__
            except Exception:
                ns = globals()

        payload = runtime._filtered_namespace(ns, include_private=include_private)
        fory = runtime.pyfory.Fory(xlang=False, ref=True, strict=strict, policy=policy)
        payload = runtime._serializable_subset(payload, fory)

        env = runtime.pack(
            payload,
            sender=self.owner or "unknown",
            name="session_state",
            strict=strict,
            policy=policy,
        )
        env.manifest["session_state"] = {
            "session_id": self.session_id,
            "peer": self.peer,
            "owner": self.owner,
            "role": self.role,
            "created_at": self.created_at,
            "saved_at": _iso_now(),
            "versions": _version_info(),
            "items": len(payload),
        }

        target.write_text(json.dumps(runtime._envelope_record(env), indent=2))
        return target

    def load(
        self,
        path: Optional[Path | str] = None,
        *,
        globals_ns: Optional[dict] = None,
        overwrite: bool = False,
        strict: bool = False,
        policy=None,
        auto_accept: bool = False,
    ):
        """
        Restore variables from a saved session state file into globals.
        """
        from . import runtime

        target = Path(path) if path is not None else self._default_state_path()
        if not target.exists():
            raise FileNotFoundError(f"State file not found: {target}")

        env = runtime.read_envelope(target)
        meta = env.manifest.get("session_state", {})

        if meta and meta.get("session_id") and meta["session_id"] != self.session_id:
            print(
                f"⚠️  State file session_id {meta.get('session_id')} does not match "
                f"active session {self.session_id}"
            )

        saved_versions = meta.get("versions", {})
        current_versions = _version_info()
        mismatches = []
        for key, saved_val in saved_versions.items():
            current_val = current_versions.get(key)
            if saved_val and current_val and saved_val != current_val:
                mismatches.append(f"{key}: saved {saved_val}, current {current_val}")
        if mismatches:
            print("⚠️  Version differences detected:")
            for line in mismatches:
                print(f"   - {line}")

        backend = (
            self._context._backend if self._context and hasattr(self._context, "_backend") else None
        )
        # Session state is user's own saved data - use trusted policy by default
        from .policy import TRUSTED_POLICY

        effective_policy = policy if policy is not None else TRUSTED_POLICY
        obj = runtime.unpack(
            env, strict=strict, policy=effective_policy, auto_accept=auto_accept, backend=backend
        )

        target_globals = globals_ns
        if target_globals is None:
            try:
                import __main__

                target_globals = __main__.__dict__
            except Exception:
                target_globals = globals()

        if target_globals is not None:
            if not overwrite:
                should = runtime._check_overwrite(
                    obj, globals_ns=target_globals, name_hint=env.name
                )
                if not should:
                    print("⚠️  Load cancelled - no variables were overwritten")
                    return obj
            injected = runtime._inject(obj, globals_ns=target_globals, name_hint=env.name)
            if injected:
                print(f"✓ Restored {len(injected)} variables from state")

        return obj

    @property
    def remote_vars(self):
        """
        Access remote variables registry for this session.

        Use this to publish Twins within the session context:
            session.remote_vars["patient_data"] = patient_twin

        Returns:
            RemoteVarRegistry scoped to this session
        """
        if self._context is None:
            raise RuntimeError("Session not bound to a context.")

        from .remote_vars import RemoteVarRegistry

        if self._local_path is None:
            self._setup_paths()

        registry_path = self._local_path / "remote_vars.json"
        return RemoteVarRegistry(
            owner=self.owner,
            registry_path=registry_path,
            session_id=self.session_id,
            context=self._context,
            peer=self.peer,  # Allow encrypting for peer
        )

    @property
    def peer_remote_vars(self):
        """
        View peer's remote variables in this session.

        Use this to access data the peer has published:
            patient = session.peer_remote_vars["patient_data"].load()

        Returns:
            RemoteVarView of peer's session vars
        """
        if self._context is None:
            raise RuntimeError("Session not bound to a context.")

        from .remote_vars import RemoteVarView

        if self._peer_path is None:
            self._setup_paths()

        registry_path = self._peer_path / "remote_vars.json"
        data_dir = self._peer_path / "data"

        return RemoteVarView(
            remote_user=self.peer,
            local_user=self.owner,
            registry_path=registry_path,
            data_dir=data_dir,
            context=self._context,
            session=self,  # Pass session reference for write path resolution
        )

    def wait_for_remote_var(
        self,
        name: str,
        *,
        timeout: float = 120.0,
        poll_interval: float = 2.0,
        load: bool = True,
        auto_accept: bool = False,
        trust_loader: bool | None = None,
    ):
        """
        Wait for a remote variable to be published by the peer.

        Args:
            name: Name of the remote variable to wait for
            timeout: Max seconds to wait (default 120)
            poll_interval: Seconds between checks (default 2)
            load: If True, load and return the value (default True)
            auto_accept: If True, automatically accept computation requests (default False)
            trust_loader: If True, run loader in trusted mode (full builtins).
                         If False, use RestrictedPython only (may fail).
                         If None (default), try RestrictedPython first and prompt on failure.

        Returns:
            The loaded value if load=True, otherwise the RemoteVarEntry

        Example:
            # Wait for peer to publish "patient_data"
            patient = session.wait_for_remote_var("patient_data")

            # With auto_accept for computation results
            result = session.wait_for_remote_var("result", auto_accept=True)
        """
        import time

        if self._context is None:
            raise RuntimeError("Session not bound to a context.")

        print(f"⏳ Waiting for '{name}' from {self.peer}...")

        announced = False
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                peer_vars = self.peer_remote_vars
                if name in peer_vars:
                    if not announced:
                        print(f"📬 '{name}' is now available!")
                        announced = True

                    entry = peer_vars[name]
                    if not load:
                        return entry

                    from .remote_vars import NotReady

                    value = entry.load(
                        inject=False,
                        auto_accept=auto_accept,
                        trust_loader=trust_loader,
                    )
                    if isinstance(value, NotReady):
                        # Data exists but isn't locally available yet (sync/decrypt pending).
                        continue

                    if value is entry:
                        if getattr(entry, "_last_error", None) is not None:
                            raise RuntimeError(
                                f"Remote var '{name}' exists but could not be loaded: {entry._last_error}"
                            )
                        # Pointer-only or metadata-only; keep waiting.
                        continue

                    # Value may legitimately be None (published NoneType); return it.
                    return value
            except RuntimeError:
                # Surface permanent load errors (deserialization policy, schema mismatch, etc.)
                raise
            except Exception:
                # Registry might not exist yet or be unreadable
                pass

            time.sleep(poll_interval)

        print(f"⏰ Timeout after {timeout}s waiting for '{name}'")
        return None

    def inbox(self):
        """
        Check peer's session folder for incoming messages.

        This looks at the peer's session folder (synced via SyftBox) for
        computation requests, results, or other messages.

        Returns:
            InboxView of peer's session messages
        """
        if self._context is None:
            raise RuntimeError("Session not bound to a context.")

        from .runtime import InboxView, list_inbox

        if self._peer_path is None:
            self._setup_paths()

        # Get backend for decryption if available
        backend = getattr(self._context, "_backend", None)

        # Look for .beaver files in peer's session folder
        envelopes = list_inbox(self._peer_path, backend=backend)
        return InboxView(self._peer_path, envelopes, session=self)

    def send(self, obj, **kwargs):
        """
        Send an object to the peer via this session.

        Files are written to our local session folder (readable by peer).

        Args:
            obj: Object to send
            **kwargs: Additional options (name, etc.)

        Returns:
            SendResult with path and envelope info
        """
        if self._context is None:
            raise RuntimeError("Session not bound to a context.")

        if self._local_path is None:
            self._setup_paths()

        from .runtime import SendResult, pack, write_envelope

        # Use session folder as destination
        name = kwargs.get("name")
        if name is None and hasattr(obj, "name"):
            name = obj.name

        # Determine backend and recipients for encryption
        # In SyftBox, data is encrypted FOR THE RECIPIENT (peer) only
        backend = None
        recipients = None
        if self._context and hasattr(self._context, "_backend") and self._context._backend:
            backend = self._context._backend
            if backend.uses_crypto:
                # Encrypt for the peer who will read this data
                recipients = [self.peer]

        # Use data/ subdirectory for all beaver files
        data_dir = self._local_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        env = pack(
            obj,
            sender=self.owner,
            name=name,
            artifact_dir=data_dir,
            backend=backend,
            recipients=recipients,
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
            path = data_dir / env.filename()
            backend.storage.write_with_shadow(
                str(path),
                content,
                recipients=recipients,
                hint="beaver-envelope",
            )
        else:
            # Write to our session data folder (peer can read it)
            path = write_envelope(env, out_dir=data_dir)
        print(f"📤 Sent to session: {path.name}")

        return SendResult(path=path, envelope=env)

    def workspace(self):
        """
        View the session workspace - shows both our data and peer's data.

        Returns:
            SessionWorkspace view of the session state
        """
        if self._context is None:
            raise RuntimeError("Session not bound to a context.")

        if self._local_path is None or self._peer_path is None:
            self._setup_paths()

        from .runtime import list_inbox

        # Get backend for decryption if available
        backend = getattr(self._context, "_backend", None)

        # Get our published data
        our_files = list_inbox(self._local_path, backend=backend)
        # Get peer's published data
        peer_files = list_inbox(self._peer_path, backend=backend)

        return SessionWorkspace(
            session=self,
            local_files=our_files,
            peer_files=peer_files,
        )

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "session_id": self.session_id,
            "peer": self.peer,
            "owner": self.owner,
            "role": self.role,
            "created_at": self.created_at,
            "accepted_at": self.accepted_at,
            "status": self.status,
            "datasets": [ds.to_dict() for ds in self.datasets],
        }

    @classmethod
    def from_dict(cls, data: dict, context: Optional[BeaverContext] = None) -> Session:
        """Create from dict."""
        # Parse datasets list
        datasets_data = data.get("datasets", [])
        datasets = [SessionDatasetInfo.from_dict(d) for d in datasets_data]

        session = cls(
            session_id=data["session_id"],
            peer=data["peer"],
            owner=data["owner"],
            role=data["role"],
            created_at=data.get("created_at", _iso_now()),
            accepted_at=data.get("accepted_at"),
            status=data.get("status", "pending"),
            datasets=datasets,
        )
        session._context = context
        if session.status == "active":
            session._setup_paths()
        return session

    def __repr__(self) -> str:
        status_icon = {"pending": "⏳", "active": "🟢", "closed": "🔴"}.get(self.status, "❓")
        lines = [
            f"{status_icon} Session with {self.peer}",
            f"   ID: {self.session_id[:12]}...",
            f"   Role: {self.role}",
            f"   Status: {self.status}",
        ]

        # Include inbox if session is active
        if self.status == "active" and self._context is not None:
            try:
                inbox_view = self.inbox()
                lines.append("")
                lines.append("📥 session.inbox():")
                # Indent inbox output
                inbox_str = repr(inbox_view)
                for line in inbox_str.split("\n"):
                    lines.append(f"   {line}")
            except Exception:
                pass  # Don't fail repr if inbox fails

        return "\n".join(lines)

    def _repr_html_(self) -> str:
        status_icon = {"pending": "⏳", "active": "🟢", "closed": "🔴"}.get(self.status, "❓")
        html = (
            f"<div style='border:1px solid #ccc; padding:10px; margin:5px; border-radius:5px;'>"
            f"<b>{status_icon} Session with {self.peer}</b><br>"
            f"<code>ID: {self.session_id[:12]}...</code><br>"
            f"Role: {self.role}<br>"
            f"Status: {self.status}"
        )

        # Include inbox if session is active
        if self.status == "active" and self._context is not None:
            try:
                inbox_view = self.inbox()
                inbox_html = inbox_view._repr_html_()
                html += f"<br><br><b>📥 session.inbox():</b><br>{inbox_html}"
            except Exception:
                pass  # Don't fail repr if inbox fails

        html += "</div>"
        return html


class SessionRequestsView:
    """Pretty-printable view of pending session requests."""

    def __init__(self, requests: list[SessionRequest]):
        self.requests = requests

    def __len__(self) -> int:
        return len(self.requests)

    def __getitem__(self, key) -> SessionRequest:
        if isinstance(key, int):
            return self.requests[key]
        elif isinstance(key, str):
            # Match by session_id or requester email
            for req in self.requests:
                if req.session_id == key or req.session_id.startswith(key):
                    return req
                if req.requester == key:
                    return req
            raise KeyError(f"No session request matching: {key}")
        raise TypeError(f"Invalid index type: {type(key)}")

    def __iter__(self):
        return iter(self.requests)

    def __repr__(self) -> str:
        if not self.requests:
            return "SessionRequests: empty"

        lines = [f"SessionRequests ({len(self.requests)} pending):"]
        for i, req in enumerate(self.requests):
            status_icon = {"pending": "⏳", "accepted": "✅", "rejected": "❌"}.get(
                req.status, "❓"
            )
            lines.append(f"  [{i}] {status_icon} {req.session_id[:8]}... from {req.requester}")
        lines.append("")
        lines.append("Use [index] or [session_id] to select, then .accept() to approve")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        if not self.requests:
            return "<b>SessionRequests</b>: empty"

        rows = []
        for i, req in enumerate(self.requests):
            status_icon = {"pending": "⏳", "accepted": "✅", "rejected": "❌"}.get(
                req.status, "❓"
            )
            rows.append(
                f"<tr>"
                f"<td>[{i}]</td>"
                f"<td>{status_icon}</td>"
                f"<td><code>{req.session_id}</code></td>"
                f"<td>{req.requester}</td>"
                f"<td>{req.created_at[:19]}</td>"
                f"</tr>"
            )

        return (
            f"<b>SessionRequests</b> ({len(self.requests)} pending)<br>"
            f"<table>"
            f"<thead><tr><th>#</th><th>Status</th><th>Session ID</th><th>From</th><th>Created</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            f"</table>"
            f"<i>Use [index] or [session_id] to select, then .accept() to approve</i>"
        )
