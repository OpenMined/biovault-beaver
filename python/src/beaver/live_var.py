"""LiveVar: Simple live-syncing variable without Twin's public/private split."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .live_mixin import LiveMixin
from .remote_data import RemoteData


@dataclass
class LiveVar(LiveMixin, RemoteData):
    """
    A simple live-syncing variable.

    Unlike Twin, LiveVar has no public/private split - it's just a value
    that can be synced in real-time. Use this when you don't need to hide
    data from one side, like progress tracking.

    Examples:
        # Create a progress tracker
        progress = LiveVar(value={"epoch": 0, "status": "waiting"}, name="progress")
        progress.enable_live(interval=0.5)
        session.remote_vars["progress"] = progress

        # Update it during training
        progress.value = {"epoch": 1, "status": "training", "acc": 0.85}

        # Peer sees updates in real-time
        p = session.peer_remote_vars["progress"].load()
        print(p.value)  # {"epoch": 1, "status": "training", "acc": 0.85}
    """

    value: Any = field(default=None)
    var_type: str = field(default="LiveVar")

    def __post_init__(self):
        """Initialize LiveMixin state."""
        super().__post_init__()
        if self.var_type == "unknown":
            self.var_type = "LiveVar"

    def has_data(self) -> bool:
        """Check if data is available."""
        return self.value is not None

    def get_value(self) -> Any:
        """Get the current value."""
        return self.value

    def request_private(self, context=None):  # noqa: ARG002
        """No private data in LiveVar - just return value."""
        return self.value

    def __repr__(self) -> str:
        """String representation."""
        id_str = self.id[:8] if self.id else "unknown"
        lines = [f"LiveVar: {self.name or f'#{id_str}'}"]

        # Show value preview
        val_str = str(self.value)
        if len(val_str) > 60:
            val_str = val_str[:60] + "..."
        lines.append(f"  Value: {val_str}")
        lines.append(f"  Owner: {self.owner}")

        # Live status
        lines.extend(self._display_live_status())

        id_display = self.id[:12] if self.id else "unknown"
        lines.append(f"  ID: {id_display}...")
        return "\n".join(lines)

    def __str__(self) -> str:
        """Use repr for string conversion."""
        return self.__repr__()
