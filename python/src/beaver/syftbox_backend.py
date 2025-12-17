"""SyftBox storage backend for Beaver.

This module provides encrypted read/write operations using the SyftBox SDK,
enabling end-to-end encryption of .beaver files exchanged between users.
"""

from __future__ import annotations

import base64
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import beaver

if TYPE_CHECKING:
    from .envelope import BeaverEnvelope
    from .session import SessionRequest


class SenderVerificationError(Exception):
    """Raised when envelope sender doesn't match cryptographically verified sender."""

    def __init__(self, claimed_sender: str, verified_sender: str):
        self.claimed_sender = claimed_sender
        self.verified_sender = verified_sender
        super().__init__(
            f"Sender verification failed: envelope claims '{claimed_sender}' "
            f"but crypto verified '{verified_sender}'"
        )


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialize_envelope(envelope: BeaverEnvelope) -> bytes:
    """Serialize a BeaverEnvelope to bytes (JSON format)."""
    record = {
        "version": envelope.version,
        "envelope_id": envelope.envelope_id,
        "sender": envelope.sender,
        "created_at": envelope.created_at,
        "name": envelope.name,
        "inputs": envelope.inputs,
        "outputs": envelope.outputs,
        "requirements": envelope.requirements,
        "manifest": envelope.manifest,
        "reply_to": envelope.reply_to,
        "payload_b64": base64.b64encode(envelope.payload).decode("ascii"),
    }
    return json.dumps(record, indent=2).encode("utf-8")


def _deserialize_envelope(data: bytes) -> BeaverEnvelope:
    """Deserialize bytes to a BeaverEnvelope."""
    from .envelope import BeaverEnvelope

    record = json.loads(data.decode("utf-8"))
    payload = base64.b64decode(record["payload_b64"])
    return BeaverEnvelope(
        version=record.get("version", 1),
        envelope_id=record.get("envelope_id"),
        sender=record.get("sender", "unknown"),
        created_at=record.get("created_at"),
        name=record.get("name"),
        inputs=record.get("inputs", []),
        outputs=record.get("outputs", []),
        requirements=record.get("requirements", []),
        manifest=record.get("manifest", {}),
        reply_to=record.get("reply_to"),
        payload=payload,
    )


class SyftBoxBackend:
    """
    Wraps SyftBox SDK for Beaver's read/write operations.

    Provides transparent encryption/decryption of .beaver files using
    the shadow folder pattern:
    - Plaintext files in unencrypted/ (local cache)
    - Encrypted files in datasites/ (synced via SyftBox)

    Args:
        data_dir: Root data directory (contains datasites/, .syc/, unencrypted/)
        email: User's email identity (e.g., "client1@sandbox.local")
        vault_path: Override .syc vault location (default: data_dir/.syc)
        disable_crypto: Disable encryption for testing (default: False)
    """

    def __init__(
        self,
        data_dir: Path | str,
        email: str,
        vault_path: Optional[Path | str] = None,
        disable_crypto: bool = False,
    ):
        try:
            import syftbox_sdk as syft
        except ImportError as e:
            raise ImportError(
                "syftbox-sdk is required for SyftBox backend. Install with: pip install syftbox-sdk"
            ) from e

        self._syft = syft
        self.data_dir = Path(data_dir).resolve()
        # Allow env override so we never fall back to "unknown"
        if email == "unknown":
            import os

            email = os.environ.get("SYFTBOX_EMAIL", email)
        self.email = email
        self.disable_crypto = disable_crypto

        # Determine vault path
        if vault_path:
            self.vault_path = Path(vault_path)
        else:
            self.vault_path = self.data_dir / ".syc"

        # Initialize SyftBox storage
        # The root should point to datasites/ directory
        datasites_root = self.data_dir / "datasites"
        datasites_root.mkdir(parents=True, exist_ok=True)

        self.storage = syft.SyftBoxStorage(
            root=str(datasites_root),
            vault_path=str(self.vault_path) if not disable_crypto else None,
            disable_crypto=disable_crypto,
            debug=beaver.debug,
        )

        # Initialize SyftBox app structure for biovault
        self.app = syft.SyftBoxApp(
            data_dir=str(self.data_dir),
            email=email,
            app_name="biovault",
        )

    @property
    def uses_crypto(self) -> bool:
        """Check if encryption is enabled."""
        return self.storage.uses_crypto()

    @property
    def shared_path(self) -> Path:
        """Path to user's shared/biovault folder (encrypted)."""
        return self.data_dir / "datasites" / self.email / "shared" / "biovault"

    @property
    def shadow_path(self) -> Path:
        """Path to decrypted shadow folder."""
        return self.data_dir / "unencrypted" / self.email / "shared" / "biovault"

    def _ensure_dirs(self) -> None:
        """Ensure required directories exist."""
        self.shared_path.mkdir(parents=True, exist_ok=True)
        self.shadow_path.mkdir(parents=True, exist_ok=True)

    def write_envelope(
        self,
        envelope: BeaverEnvelope,
        recipients: list[str],
        filename: Optional[str] = None,
    ) -> Path:
        """
        Write encrypted envelope to shared folder.

        Args:
            envelope: BeaverEnvelope to write
            recipients: List of recipient email addresses for encryption
            filename: Override filename (default: envelope_id.beaver)

        Returns:
            Path to the written file in datasites/
        """
        self._ensure_dirs()

        filename = filename or f"{envelope.envelope_id}.beaver"

        # Full path in datasites
        datasite_path = self.shared_path / filename

        # Serialize envelope to bytes
        data = _serialize_envelope(envelope)

        if self.uses_crypto and recipients:
            # Write with encryption using shadow pattern
            self.storage.write_with_shadow(
                absolute_path=str(datasite_path),
                data=data,
                recipients=recipients,
                hint="beaver-envelope",
                overwrite=True,
            )
        else:
            # Plaintext write (no encryption)
            self.storage.write_bytes(str(datasite_path), data, overwrite=True)

        return datasite_path

    def read_envelope(self, path: Path | str) -> BeaverEnvelope:
        """
        Read and decrypt envelope from any datasite.

        Args:
            path: Path to .beaver file (in datasites/)

        Returns:
            Decrypted BeaverEnvelope
        """
        path = Path(path)

        if self.uses_crypto:
            # Read with decryption using shadow pattern
            data = bytes(self.storage.read_with_shadow(str(path)))
        else:
            # Plaintext read
            data = bytes(self.storage.read_bytes(str(path)))

        env = _deserialize_envelope(data)
        # Non-serialized hint used to resolve TrustedLoader relative artifact paths.
        env._path = str(path)  # type: ignore[attr-defined]
        return env

    def read_envelope_verified(
        self, path: Path | str, verify: bool = True
    ) -> tuple[BeaverEnvelope, str, str]:
        """
        Read envelope and verify sender identity cryptographically.

        Uses syftbox-sdk's read_with_shadow_metadata to get the verified sender
        from the encrypted envelope's signature, then compares it to the
        claimed sender in the envelope.

        Args:
            path: Path to .beaver file (in datasites/)
            verify: If True, raise SenderVerificationError on mismatch.
                   If False, just return the verified sender for inspection.

        Returns:
            Tuple of (envelope, verified_sender, fingerprint)
            - envelope: Decrypted BeaverEnvelope
            - verified_sender: Cryptographically verified sender identity
            - fingerprint: Sender's identity key fingerprint (SHA256 hex)

        Raises:
            SenderVerificationError: If verify=True and claimed sender doesn't
                                    match verified sender
        """
        path = Path(path)

        if self.uses_crypto:
            # Read with metadata to get verified sender
            result = self.storage.read_with_shadow_metadata(str(path))
            data = bytes(result.data)
            verified_sender = result.sender
            fingerprint = result.fingerprint
        else:
            # Plaintext read - no crypto verification possible
            data = bytes(self.storage.read_bytes(str(path)))
            verified_sender = "(plaintext)"
            fingerprint = "(none)"

        envelope = _deserialize_envelope(data)
        # Non-serialized hint used to resolve TrustedLoader relative artifact paths.
        envelope._path = str(path)  # type: ignore[attr-defined]

        # Verify sender if crypto is enabled and file was encrypted
        if (
            verify
            and verified_sender not in ("(plaintext)", "(none)")
            and envelope.sender != verified_sender
        ):
            raise SenderVerificationError(envelope.sender, verified_sender)

        return envelope, verified_sender, fingerprint

    def write_bytes(
        self,
        relative_path: str,
        data: bytes,
        recipients: Optional[list[str]] = None,
    ) -> Path:
        """
        Write bytes to user's shared folder.

        Args:
            relative_path: Path relative to shared/biovault/
            data: Bytes to write
            recipients: Optional list of recipients for encryption

        Returns:
            Full path to written file
        """
        self._ensure_dirs()
        full_path = self.shared_path / relative_path

        if self.uses_crypto and recipients:
            self.storage.write_with_shadow(
                absolute_path=str(full_path),
                data=data,
                recipients=recipients,
                hint=None,
                overwrite=True,
            )
        else:
            self.storage.write_bytes(str(full_path), data, overwrite=True)

        return full_path

    def read_bytes(self, path: Path | str) -> bytes:
        """
        Read bytes from any path (with decryption if needed).

        Args:
            path: Path to file

        Returns:
            Decrypted bytes
        """
        path = Path(path)

        if self.uses_crypto:
            return bytes(self.storage.read_with_shadow(str(path)))
        else:
            return bytes(self.storage.read_bytes(str(path)))

    def list_envelopes(self, peer_email: Optional[str] = None) -> list[Path]:
        """
        List .beaver files in a shared folder.

        Args:
            peer_email: Peer's email (default: own email)

        Returns:
            List of paths to .beaver files
        """
        email = peer_email or self.email
        peer_shared = self.data_dir / "datasites" / email / "shared" / "biovault"

        if not peer_shared.exists():
            return []

        return sorted(peer_shared.glob("*.beaver"))

    def list_peer_envelopes(self, peer_email: str) -> list[Path]:
        """
        List .beaver files in peer's shared folder.

        Args:
            peer_email: Peer's email address

        Returns:
            List of paths to .beaver files
        """
        return self.list_envelopes(peer_email)

    def path_exists(self, path: Path | str) -> bool:
        """Check if a path exists."""
        return self.storage.path_exists(str(path))

    def remove_path(self, path: Path | str) -> None:
        """Remove a file (and its shadow copy)."""
        self.storage.remove_path(str(path))

    def import_peer_bundle(self, peer_email: str, bundle_path: Path | str) -> str:
        """
        Import a peer's public crypto bundle into the vault.

        Args:
            peer_email: Expected peer identity
            bundle_path: Path to peer's did.json bundle

        Returns:
            Verified peer identity
        """
        return self._syft.import_bundle(
            bundle_path=str(bundle_path),
            vault_path=str(self.vault_path),
            expected_identity=peer_email,
        )

    def get_peer_inbox_path(self, peer_email: str) -> Path:
        """Get path to peer's shared/biovault folder."""
        return self.data_dir / "datasites" / peer_email / "shared" / "biovault"

    def ensure_peer_datasite(self, peer_email: str) -> None:
        """Ensure peer's datasite directory exists."""
        peer_path = self.get_peer_inbox_path(peer_email)
        peer_path.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Session RPC Methods
    # -------------------------------------------------------------------------

    def get_rpc_session_path(self) -> Path:
        """Get path to session RPC folder."""
        return (
            self.data_dir / "datasites" / self.email / "app_data" / "biovault" / "rpc" / "session"
        )

    def get_peer_rpc_session_path(self, peer_email: str) -> Path:
        """Get path to peer's session RPC folder."""
        return (
            self.data_dir / "datasites" / peer_email / "app_data" / "biovault" / "rpc" / "session"
        )

    def _wait_for_peer_rpc_acl(
        self, peer_email: str, timeout: int = 30, poll_interval: float = 1.0
    ) -> bool:
        """
        Wait until the peer's RPC ACL (syft.pub.yaml) is visible before we try to write.
        This avoids issuing writes that the relay will reject because the ACL has not synced yet.
        """
        rpc_dir = self.get_peer_rpc_session_path(peer_email).parent
        # Look for either the rpc-level ACL or the app-level ACL
        candidates = [
            rpc_dir / "syft.pub.yaml",
            rpc_dir.parent / "syft.pub.yaml",
        ]
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for path in candidates:
                try:
                    if path.exists():
                        return True
                except OSError:
                    # Keep trying until timeout if the path is temporarily inaccessible
                    pass
            time.sleep(poll_interval)
        return False

    def send_session_request(
        self,
        session_id: str,
        target_email: str,
        message: Optional[str] = None,
    ) -> Path:
        """
        Send a session request to a peer via RPC.

        Args:
            session_id: The session ID
            target_email: Target user's email
            message: Optional message for the request

        Returns:
            Path to the request file
        """
        # Write request to peer's RPC folder
        peer_rpc = self.get_peer_rpc_session_path(target_email)
        peer_rpc.mkdir(parents=True, exist_ok=True)

        if not self._wait_for_peer_rpc_acl(target_email):
            raise TimeoutError(
                f"Peer RPC ACL not found for {target_email}; aborting session request to avoid rejection"
            )

        request_data = {
            "session_id": session_id,
            "requester": self.email,
            "target": target_email,
            "created_at": _iso_now(),
            "message": message,
            "status": "pending",
        }

        request_file = peer_rpc / f"{session_id}.request"

        if self.uses_crypto:
            # Encrypt the request for the target
            data = json.dumps(request_data, indent=2).encode("utf-8")
            self.storage.write_with_shadow(
                absolute_path=str(request_file),
                data=data,
                recipients=[target_email],
                hint="session-request",
                overwrite=True,
            )
        else:
            request_file.write_text(json.dumps(request_data, indent=2))

        return request_file

    def list_session_requests(self) -> list[SessionRequest]:
        """
        List pending session requests in our RPC folder.

        Returns:
            List of SessionRequest objects
        """
        from .session import SessionRequest

        rpc_path = self.get_rpc_session_path()
        if not rpc_path.exists():
            return []

        requests = []
        for req_file in rpc_path.glob("*.request"):
            try:
                if self.uses_crypto:
                    data = bytes(self.storage.read_with_shadow(str(req_file)))
                    req_data = json.loads(data.decode("utf-8"))
                else:
                    req_data = json.loads(req_file.read_text())

                # Only include pending requests
                if req_data.get("status") == "pending":
                    requests.append(SessionRequest.from_dict(req_data))
            except Exception as e:
                print(f"Warning: Could not read session request {req_file}: {e}")

        return requests

    def send_session_response(
        self,
        session_id: str,
        requester_email: str,
        accepted: bool,
        reason: Optional[str] = None,
    ) -> Path:
        """
        Send session acceptance/rejection response.

        Args:
            session_id: The session ID
            requester_email: Requester's email
            accepted: Whether to accept
            reason: Optional reason (for rejection)

        Returns:
            Path to the response file
        """
        # Write response to requester's RPC folder
        requester_rpc = self.get_peer_rpc_session_path(requester_email)
        requester_rpc.mkdir(parents=True, exist_ok=True)

        if not self._wait_for_peer_rpc_acl(requester_email):
            raise TimeoutError(
                f"Peer RPC ACL not found for {requester_email}; aborting session response to avoid rejection"
            )

        response_data = {
            "session_id": session_id,
            "status": "accepted" if accepted else "rejected",
            "accepted_at": _iso_now() if accepted else None,
            "rejected_at": None if accepted else _iso_now(),
            "reason": reason,
            "responder": self.email,
        }

        response_file = requester_rpc / f"{session_id}.response"

        if self.uses_crypto:
            data = json.dumps(response_data, indent=2).encode("utf-8")
            self.storage.write_with_shadow(
                absolute_path=str(response_file),
                data=data,
                recipients=[requester_email],
                hint="session-response",
                overwrite=True,
            )
        else:
            response_file.write_text(json.dumps(response_data, indent=2))

        # Also update the original request status
        our_rpc = self.get_rpc_session_path()
        request_file = our_rpc / f"{session_id}.request"
        if request_file.exists():
            try:
                if self.uses_crypto:
                    req_data = json.loads(
                        bytes(self.storage.read_with_shadow(str(request_file))).decode("utf-8")
                    )
                else:
                    req_data = json.loads(request_file.read_text())

                req_data["status"] = "accepted" if accepted else "rejected"

                if self.uses_crypto:
                    self.storage.write_with_shadow(
                        absolute_path=str(request_file),
                        data=json.dumps(req_data, indent=2).encode("utf-8"),
                        recipients=[requester_email],
                        hint="session-request",
                        overwrite=True,
                    )
                else:
                    request_file.write_text(json.dumps(req_data, indent=2))
            except Exception:
                pass

        return response_file

    def create_session_folder(
        self,
        session_id: str,
        peer_email: str,
    ) -> Path:
        """
        Create a session folder with proper permissions.

        Args:
            session_id: The session ID
            peer_email: Peer's email (for permissions)

        Returns:
            Path to the created session folder
        """
        # Use syftbox-sdk to grant peer access under the session path (owner write/read/admin)
        relative = Path("shared").joinpath("biovault").joinpath("sessions").joinpath(session_id)
        owner_path = Path(
            self.app.ensure_peer_can_read(str(relative), peer_email, allow_write=True)
        )

        # Ensure a data subfolder exists
        (owner_path / "data").mkdir(parents=True, exist_ok=True)

        # Also ensure peer side exists with matching ACL
        peer_path = (
            Path(self.data_dir).joinpath("datasites").joinpath(peer_email).joinpath(relative)
        )
        peer_path.mkdir(parents=True, exist_ok=True)
        (peer_path / "data").mkdir(parents=True, exist_ok=True)
        self.app.ensure_peer_can_read(str(relative), peer_email, allow_write=True)

        return owner_path

    def get_session_path(self, session_id: str) -> Path:
        """Get path to our session folder."""
        return (
            self.data_dir
            / "datasites"
            / self.email
            / "shared"
            / "biovault"
            / "sessions"
            / session_id
        )

    def get_peer_session_path(self, peer_email: str, session_id: str) -> Path:
        """Get path to peer's session folder."""
        return (
            self.data_dir
            / "datasites"
            / peer_email
            / "shared"
            / "biovault"
            / "sessions"
            / session_id
        )


def provision_identity(
    data_dir: Path | str,
    email: str,
    vault_path: Optional[Path | str] = None,
) -> dict:
    """
    Provision a crypto identity for a user.

    Creates private key and public bundle if they don't exist.

    Args:
        data_dir: Root data directory
        email: User's email identity
        vault_path: Override .syc vault location

    Returns:
        Dict with:
            - identity: str - The email identity
            - generated: bool - True if newly generated
            - recovery_mnemonic: Optional[str] - 12-word recovery phrase (only if generated)
            - vault_path: str - Path to .syc vault
            - bundle_path: str - Path to private bundle
            - public_bundle_path: str - Path to exported public DID
    """
    try:
        import syftbox_sdk as syft
    except ImportError as e:
        raise ImportError(
            "syftbox-sdk is required for identity provisioning. "
            "Install with: pip install syftbox-sdk"
        ) from e

    result = syft.provision_identity(
        identity=email,
        data_root=str(data_dir),
        vault_override=str(vault_path) if vault_path else None,
    )

    return {
        "identity": result.identity,
        "generated": result.generated,
        "recovery_mnemonic": result.recovery_mnemonic,
        "vault_path": result.vault_path,
        "bundle_path": result.bundle_path,
        "public_bundle_path": result.public_bundle_path,
    }


def import_peer_bundle(
    bundle_path: Path | str,
    vault_path: Path | str,
    expected_identity: Optional[str] = None,
) -> str:
    """
    Import a peer's public bundle into the local vault.

    Args:
        bundle_path: Path to peer's did.json bundle
        vault_path: Path to local .syc vault
        expected_identity: Expected peer email (validates bundle)

    Returns:
        Verified peer identity string
    """
    try:
        import syftbox_sdk as syft
    except ImportError as e:
        raise ImportError(
            "syftbox-sdk is required for bundle import. Install with: pip install syftbox-sdk"
        ) from e

    return syft.import_bundle(
        bundle_path=str(bundle_path),
        vault_path=str(vault_path),
        expected_identity=expected_identity,
    )
