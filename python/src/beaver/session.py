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
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import uuid4

if TYPE_CHECKING:
    from .runtime import BeaverContext


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generate_session_id() -> str:
    """Generate a short random session ID."""
    return hashlib.sha256(uuid4().bytes).hexdigest()[:12]


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
    _context: Optional["BeaverContext"] = field(default=None, repr=False)

    def accept(self) -> "Session":
        """
        Accept this session request.

        Creates the session folders and sends acceptance response.

        Returns:
            Session object for the accepted session
        """
        if self._context is None:
            raise RuntimeError("SessionRequest not bound to a context. Use bv.accept_session() instead.")

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
    def from_dict(cls, data: dict, context: Optional["BeaverContext"] = None) -> "SessionRequest":
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

    # Internal references
    _context: Optional["BeaverContext"] = field(default=None, repr=False)
    _local_path: Optional[Path] = field(default=None, repr=False)
    _peer_path: Optional[Path] = field(default=None, repr=False)

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

        print(f"⏰ Timeout waiting for session acceptance")
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
            except Exception as e:
                # Could be decryption error, file not fully synced, etc.
                pass

        return False

    def _setup_paths(self) -> None:
        """Setup local and peer folder paths."""
        if self._context is None or self._context._backend is None:
            return

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

        # Create our local folder
        self._local_path.mkdir(parents=True, exist_ok=True)

        # Create permission file (syft.pub.yaml)
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

        # Look for .beaver files in peer's session folder
        envelopes = list_inbox(self._peer_path)
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

        env = pack(
            obj,
            sender=self.owner,
            name=name,
            artifact_dir=self._local_path,
            backend=backend,
            recipients=recipients,
        )

        # Write envelope (encrypted if backend available)
        if backend and backend.uses_crypto and recipients:
            import json
            import base64
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
            self._local_path.mkdir(parents=True, exist_ok=True)
            path = self._local_path / env.filename()
            backend.storage.write_with_shadow(
                str(path),
                content,
                recipients=recipients,
                hint="beaver-envelope",
            )
        else:
            # Write to our session folder (peer can read it)
            path = write_envelope(env, out_dir=self._local_path)
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

        # Get our published data
        our_files = list_inbox(self._local_path)
        # Get peer's published data
        peer_files = list_inbox(self._peer_path)

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
        }

    @classmethod
    def from_dict(cls, data: dict, context: Optional["BeaverContext"] = None) -> "Session":
        """Create from dict."""
        session = cls(
            session_id=data["session_id"],
            peer=data["peer"],
            owner=data["owner"],
            role=data["role"],
            created_at=data.get("created_at", _iso_now()),
            accepted_at=data.get("accepted_at"),
            status=data.get("status", "pending"),
        )
        session._context = context
        if session.status == "active":
            session._setup_paths()
        return session

    def __repr__(self) -> str:
        status_icon = {"pending": "⏳", "active": "🟢", "closed": "🔴"}.get(self.status, "❓")
        return (
            f"{status_icon} Session(\n"
            f"    id={self.session_id!r},\n"
            f"    peer={self.peer!r},\n"
            f"    role={self.role!r},\n"
            f"    status={self.status!r}\n"
            f")"
        )

    def _repr_html_(self) -> str:
        status_icon = {"pending": "⏳", "active": "🟢", "closed": "🔴"}.get(self.status, "❓")
        return (
            f"<div style='border:1px solid #ccc; padding:10px; margin:5px; border-radius:5px;'>"
            f"<b>{status_icon} Session</b><br>"
            f"<code>ID: {self.session_id}</code><br>"
            f"Peer: <b>{self.peer}</b><br>"
            f"Role: {self.role}<br>"
            f"Status: {self.status}"
            f"</div>"
        )


class SessionRequestsView:
    """Pretty-printable view of pending session requests."""

    def __init__(self, requests: List[SessionRequest]):
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
            status_icon = {"pending": "⏳", "accepted": "✅", "rejected": "❌"}.get(req.status, "❓")
            lines.append(
                f"  [{i}] {status_icon} {req.session_id[:8]}... from {req.requester}"
            )
        lines.append("")
        lines.append("Use [index] or [session_id] to select, then .accept() to approve")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        if not self.requests:
            return "<b>SessionRequests</b>: empty"

        rows = []
        for i, req in enumerate(self.requests):
            status_icon = {"pending": "⏳", "accepted": "✅", "rejected": "❌"}.get(req.status, "❓")
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
