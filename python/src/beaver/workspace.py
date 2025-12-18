"""Workspace view for unified shared state visualization."""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class WorkspaceItem:
    """A single item in the workspace (Twin, Action, etc)."""

    name: str
    owner: str
    item_type: str  # "Twin", "Action", "RemoteVar", etc.
    underlying_type: Optional[str] = None  # e.g., "DataFrame", "int"
    status: str = "unknown"  # "live", "received", "sent", "pending", "private"
    last_action: str = "unknown"
    timestamp: Optional[datetime] = None
    envelope_id: Optional[str] = None
    is_live: bool = False
    is_private: bool = False
    version_count: int = 1

    @property
    def status_icon(self) -> str:
        """Get emoji for status."""
        icons = {
            "live": "🟢",
            "received": "📥",
            "sent": "📤",
            "pending": "⏳",
            "approved": "✅",
            "denied": "❌",
            "private": "🔐",
        }
        return icons.get(self.status, "📍")

    @property
    def type_icon(self) -> str:
        """Get emoji for type."""
        if self.item_type == "Twin":
            return "📊"
        elif self.item_type == "Action":
            return "⚡"
        elif self.item_type == "RemoteVar":
            return "📡"
        else:
            return "📦"


class WorkspaceView:
    """
    Unified view of shared workspace between users.

    Shows all data/actions shared between users with status, history, and filters.
    """

    def __init__(self, context):
        """
        Initialize workspace view.

        Args:
            context: BeaverContext
        """
        self.context = context
        self._items: dict[str, WorkspaceItem] = {}

    def refresh(self):
        """Scan and rebuild workspace from inbox, outbox, and remote vars."""
        self._items = {}

        # Scan inbox
        self._scan_inbox()

        # Scan published remote vars (ours)
        self._scan_our_remote_vars()

        # Scan peer remote vars
        self._scan_peer_remote_vars()

        # Scan sent items (outbox/other user inboxes)
        self._scan_sent_items()

        # TODO: Scan outbox for sent items
        # TODO: Scan staged computations

    def _scan_inbox(self):
        """Scan inbox for received items."""
        if not self.context.inbox_path.exists():
            return

        for path in sorted(self.context.inbox_path.glob("*.beaver")):
            try:
                # Read envelope metadata
                data = json.loads(path.read_text())
                name = data.get("name", "(unnamed)")
                sender = data.get("sender", "unknown")
                created_at = data.get("created_at")
                envelope_id = data.get("envelope_id")

                # Get type from manifest
                manifest = data.get("manifest", {})
                item_type = manifest.get("type", "unknown")

                # Determine if it's a Twin
                is_twin = "Twin" in item_type or item_type.startswith("Twin[")
                underlying_type = None
                if is_twin:
                    # Extract underlying type: Twin[DataFrame] -> DataFrame
                    underlying_type = item_type.replace("Twin[", "").replace("]", "")
                    item_type = "Twin"

                # Parse timestamp
                timestamp = None
                if created_at:
                    with contextlib.suppress(Exception):
                        timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

                # Create or update item
                key = f"{sender}:{name}"
                if key in self._items:
                    # Update existing (track versions)
                    self._items[key].version_count += 1
                    if timestamp and (
                        not self._items[key].timestamp or timestamp > self._items[key].timestamp
                    ):
                        self._items[key].timestamp = timestamp
                        self._items[key].envelope_id = envelope_id
                else:
                    # New item
                    self._items[key] = WorkspaceItem(
                        name=name,
                        owner=sender,
                        item_type=item_type,
                        underlying_type=underlying_type,
                        status="received",
                        last_action="Received",
                        timestamp=timestamp,
                        envelope_id=envelope_id,
                    )

            except Exception:
                # Skip malformed files
                pass

    def _scan_our_remote_vars(self):
        """Scan our published remote vars."""
        for name, var in self.context.remote_vars.vars.items():
            key = f"{self.context.user}:{name}"

            # Determine type
            item_type = "RemoteVar"
            underlying_type = None
            if var.var_type.startswith("Twin["):
                item_type = "Twin"
                underlying_type = var.var_type.replace("Twin[", "").replace("]", "")

            # Parse timestamp
            timestamp = None
            if var.updated_at:
                with contextlib.suppress(Exception):
                    timestamp = datetime.fromisoformat(var.updated_at.replace("Z", "+00:00"))

            self._items[key] = WorkspaceItem(
                name=name,
                owner=self.context.user,
                item_type=item_type,
                underlying_type=underlying_type,
                status="sent" if var.data_location else "private",
                last_action="Published" if var.data_location else "Private",
                timestamp=timestamp,
                is_private=not var.data_location,
            )

    def _scan_peer_remote_vars(self):
        """Scan peer published remote vars."""
        # Get all peer directories
        if not self.context._public_dir.exists():
            return

        for user_dir in self.context._public_dir.iterdir():
            if not user_dir.is_dir():
                continue

            username = user_dir.name
            if username == self.context.user:
                continue  # Skip ourselves

            # Load their registry
            registry_path = user_dir / "remote_vars.json"
            if not registry_path.exists():
                continue

            try:
                data = json.loads(registry_path.read_text())
                for name, var_data in data.items():
                    key = f"{username}:{name}"

                    # Skip if we already have this from inbox
                    if key in self._items:
                        continue

                    var_type = var_data.get("var_type", "unknown")
                    item_type = "RemoteVar"
                    underlying_type = None

                    if var_type.startswith("Twin["):
                        item_type = "Twin"
                        underlying_type = var_type.replace("Twin[", "").replace("]", "")

                    # Parse timestamp
                    timestamp = None
                    updated_at = var_data.get("updated_at")
                    if updated_at:
                        with contextlib.suppress(Exception):
                            timestamp = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))

                    self._items[key] = WorkspaceItem(
                        name=name,
                        owner=username,
                        item_type=item_type,
                        underlying_type=underlying_type,
                        status="live" if var_data.get("data_location") else "private",
                        last_action="Available" if var_data.get("data_location") else "Private",
                        timestamp=timestamp,
                    )

            except Exception:
                # Skip malformed registries
                pass

    def _scan_sent_items(self):
        """Scan other users' inboxes for items we sent."""
        base = self.context._base_dir
        if not base.exists():
            return

        for user_dir in base.iterdir():
            if not user_dir.is_dir():
                continue
            # Skip our own inbox and shared/public
            if user_dir.name in {self.context.user, "shared"}:
                continue

            for path in sorted(user_dir.glob("*.beaver")):
                with contextlib.suppress(Exception):
                    data = json.loads(path.read_text())
                    if data.get("sender") != self.context.user:
                        continue

                    name = data.get("name", "(unnamed)")
                    envelope_id = data.get("envelope_id")
                    created_at = data.get("created_at")
                    manifest = data.get("manifest", {})
                    item_type = manifest.get("type", "unknown")

                    # Normalize Twin type
                    underlying_type = None
                    if "Twin" in item_type or item_type.startswith("Twin["):
                        underlying_type = item_type.replace("Twin[", "").replace("]", "")
                        item_type = "Twin"

                    timestamp = None
                    if created_at:
                        with contextlib.suppress(Exception):
                            timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

                    key = f"{self.context.user}:{name}"
                    if key in self._items:
                        # Update existing entry with fresher timestamp
                        existing = self._items[key]
                        if timestamp and (not existing.timestamp or timestamp > existing.timestamp):
                            existing.timestamp = timestamp
                            existing.envelope_id = envelope_id
                            existing.status = "sent"
                            existing.last_action = "Sent"
                            if underlying_type:
                                existing.underlying_type = underlying_type
                        existing.version_count += 1
                    else:
                        self._items[key] = WorkspaceItem(
                            name=name,
                            owner=self.context.user,
                            item_type=item_type,
                            underlying_type=underlying_type,
                            status="sent",
                            last_action="Sent",
                            timestamp=timestamp,
                            envelope_id=envelope_id,
                        )

    def __call__(
        self,
        *,
        user: Optional[str] = None,
        status: Optional[str] = None,
        item_type: Optional[str] = None,
    ) -> WorkspaceView:
        """
        Filter workspace view.

        Args:
            user: Filter by owner username
            status: Filter by status (live, received, sent, etc)
            item_type: Filter by type (Twin, Action, etc)

        Returns:
            Filtered WorkspaceView
        """
        # Refresh before filtering
        self.refresh()

        # Apply filters
        filtered = WorkspaceView(self.context)
        for key, item in self._items.items():
            if user and item.owner != user:
                continue
            if status and item.status != status:
                continue
            if item_type and item.item_type != item_type:
                continue

            filtered._items[key] = item

        return filtered

    def __getitem__(self, name: str) -> Any:
        """
        Load an item by name.

        Returns the latest version if multiple exist.
        """
        # Refresh to get latest
        self.refresh()

        # Find items with this name
        matches = [item for item in self._items.values() if item.name == name]

        if not matches:
            raise KeyError(f"Item not found: {name}")

        # Get most recent
        latest = max(
            matches, key=lambda x: x.timestamp or datetime.min.replace(tzinfo=timezone.utc)
        )

        # Load based on type
        if latest.status == "received":
            # Load from inbox
            for envelope in self.context.inbox():
                if envelope.name == name and envelope.sender == latest.owner:
                    return envelope.load(inject=False)

        elif latest.owner != self.context.user:
            # Load from peer's remote vars
            return self.context.peer(latest.owner).remote_vars[name]

        raise KeyError(f"Could not load item: {name}")

    def __repr__(self) -> str:
        """Display workspace as table."""
        # Refresh before display
        self.refresh()

        if not self._items:
            return "Workspace: empty"

        lines = []
        lines.append("━" * 80)
        lines.append(f"🗂️  Shared Workspace: {self.context.user}")
        lines.append("━" * 80)

        # Header
        lines.append(f"{'Name':<20} {'Owner':<10} {'Type':<15} {'Status':<12} {'Last Action':<20}")
        lines.append("─" * 80)

        # Sort by timestamp (newest first)
        sorted_items = sorted(
            self._items.values(),
            key=lambda x: x.timestamp or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )

        # Limit to max_items
        max_items = self.context.settings.workspace_max_items
        for item in sorted_items[:max_items]:
            # Skip private items unless configured to show
            if item.is_private and not self.context.settings.workspace_show_private:
                continue

            # Format type
            type_str = item.item_type
            if item.underlying_type:
                type_str = f"{item.item_type}[{item.underlying_type}]"

            # Format name (with version count if > 1)
            name_str = item.name
            if item.version_count > 1:
                name_str = f"{name_str} (v{item.version_count})"

            # Format last action with time ago
            last_action = item.last_action
            if item.timestamp:
                now = datetime.now(timezone.utc)
                delta = now - item.timestamp
                if delta.days > 0:
                    time_ago = f"{delta.days}d ago"
                elif delta.seconds >= 3600:
                    time_ago = f"{delta.seconds // 3600}h ago"
                elif delta.seconds >= 60:
                    time_ago = f"{delta.seconds // 60}m ago"
                else:
                    time_ago = f"{delta.seconds}s ago"
                last_action = time_ago

            # Format status with icon
            status_str = f"{item.status_icon} {item.status.title()}"

            lines.append(
                f"{name_str:<20} {item.owner:<10} {type_str:<15} {status_str:<12} {last_action:<20}"
            )

        if len(sorted_items) > max_items:
            lines.append(
                f"... and {len(sorted_items) - max_items} more (increase bv.settings.workspace_max_items)"
            )

        lines.append("━" * 80)
        lines.append("💡 Access: ws['name'], Filter: ws(user='bob'), ws(status='live')")
        lines.append("━" * 80)

        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter."""
        # Refresh before display
        self.refresh()

        if not self._items:
            return "<b>Workspace</b>: empty"

        # Sort by timestamp (newest first)
        sorted_items = sorted(
            self._items.values(),
            key=lambda x: x.timestamp or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )

        rows = []
        max_items = self.context.settings.workspace_max_items
        for item in sorted_items[:max_items]:
            # Skip private items unless configured to show
            if item.is_private and not self.context.settings.workspace_show_private:
                continue

            # Format type
            type_str = item.item_type
            if item.underlying_type:
                type_str = f"{item.item_type}[{item.underlying_type}]"

            # Format name
            name_str = item.name
            if item.version_count > 1:
                name_str = f"{name_str} <small>(v{item.version_count})</small>"

            # Format last action
            last_action = item.last_action
            if item.timestamp:
                now = datetime.now(timezone.utc)
                delta = now - item.timestamp
                if delta.days > 0:
                    time_ago = f"{delta.days}d ago"
                elif delta.seconds >= 3600:
                    time_ago = f"{delta.seconds // 3600}h ago"
                elif delta.seconds >= 60:
                    time_ago = f"{delta.seconds // 60}m ago"
                else:
                    time_ago = f"{delta.seconds}s ago"
                last_action = time_ago

            # Status with icon
            status_str = f"{item.status_icon} {item.status.title()}"

            rows.append(
                f"<tr style='background: #111; color: #eee; border-bottom: 1px solid #222; text-align: left;'>"
                f"<td style='padding: 8px 12px; word-break: break-word;'><b>{name_str}</b></td>"
                f"<td style='padding: 8px 12px;'>{item.owner}</td>"
                f"<td style='padding: 8px 12px; font-family: monospace;'>{type_str}</td>"
                f"<td style='padding: 8px 12px;'>{status_str}</td>"
                f"<td style='padding: 8px 12px;'>{last_action}</td>"
                f"</tr>"
            )

        return (
            f"<b>🗂️ Shared Workspace: {self.context.user}</b><br>"
            f"<table style='border-collapse: collapse; table-layout: fixed; width: 100%; background: #0d0d0d; color: #eee; border: 1px solid #333;'>"
            f"<thead><tr style='background: #1d1d1d; color: #fafafa; text-align: left;'>"
            f"<th style='padding: 10px 12px;'>Name</th>"
            f"<th style='padding: 10px 12px;'>Owner</th>"
            f"<th style='padding: 10px 12px;'>Type</th>"
            f"<th style='padding: 10px 12px;'>Status</th>"
            f"<th style='padding: 10px 12px;'>Last Action</th>"
            f"</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            f"</table>"
        )
