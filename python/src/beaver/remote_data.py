"""RemoteData: Base class for all remote data references."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_current_owner() -> str:
    """
    Detect the current owner/user from environment or config.

    Checks in order:
    1. SYFTBOX_EMAIL environment variable
    2. session.json in current working directory
    3. Falls back to "unknown"

    Returns:
        The current user's email/identity
    """
    # 1. Check SYFTBOX_EMAIL env var (set by desktop when launching Jupyter)
    email = os.environ.get("SYFTBOX_EMAIL")
    if email:
        return email

    # 2. Check session.json in cwd
    session_json_path = Path(os.getcwd()) / "session.json"
    if session_json_path.exists():
        try:
            import json

            with open(session_json_path) as f:
                config = json.load(f)
                owner = config.get("owner")
                if owner:
                    return owner
        except Exception:
            pass

    return "unknown"


@dataclass
class RemoteData(ABC):
    """
    Base class for all remote data references.

    Provides identity, metadata, and standard interface for accessing data
    whether it's local, remote, or privacy-protected.
    """

    # Identity
    id: str = field(default_factory=lambda: uuid4().hex)
    name: Optional[str] = None
    owner: str = field(default_factory=get_current_owner)

    # Metadata
    var_type: str = "unknown"
    created_at: str = field(default_factory=_iso_now)
    metadata: dict = field(default_factory=dict)

    @abstractmethod
    def has_data(self) -> bool:
        """Check if data is available locally."""
        ...

    @abstractmethod
    def get_value(self) -> Any:
        """Get the data value."""
        ...

    @abstractmethod
    def request_private(self, context=None):
        """Request access to private data from owner."""
        ...

    def _display_header(self) -> str:
        """Generate common header for display."""
        var_name = self.name or f"#{self.id[:8]}"
        return f"{self.__class__.__name__}: {var_name}"

    def _display_metadata(self) -> list[str]:
        """Generate common metadata lines for display."""
        lines = []
        lines.append(f"  Type: {self.var_type}")
        lines.append(f"  Owner: {self.owner}")

        # Add size/shape if available
        if "size_bytes" in self.metadata:
            size = self.metadata["size_bytes"]
            if size < 1024:
                lines.append(f"  Size: {size} bytes")
            elif size < 1024 * 1024:
                lines.append(f"  Size: {size / 1024:.1f} KB")
            else:
                lines.append(f"  Size: {size / (1024 * 1024):.1f} MB")

        if "shape" in self.metadata:
            lines.append(f"  Shape: {self.metadata['shape']}")

        return lines

    def _display_ids(self) -> str:
        """Generate ID line for display."""
        return f"  ID: {self.id[:12]}..."

    def __str__(self) -> str:
        """Default string representation."""
        lines = [self._display_header()]
        lines.extend(self._display_metadata())
        lines.append(self._display_ids())
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Use string representation."""
        return self.__str__()
