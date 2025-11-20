"""Settings management for BeaverContext."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class BeaverSettings:
    """
    Configuration settings for Beaver data sharing.

    Settings can be modified at runtime and persist for the session.
    """

    # Inbox retention settings
    inbox_max_versions: int = 10
    """Maximum number of versions to keep for objects with the same name (default: 10)"""

    inbox_retention_hours: Optional[float] = 24.0
    """Delete versions older than this many hours (None = keep forever, default: 24)"""

    inbox_auto_cleanup: bool = True
    """Automatically clean up old versions when loading inbox (default: True)"""

    # Live sync settings
    live_default_interval: float = 2.0
    """Default sync interval in seconds for live-enabled Twins (default: 2.0)"""

    live_auto_subscribe: bool = True
    """Auto-subscribe to live updates when loading live-enabled Twins (default: True)"""

    # Workspace settings
    workspace_max_items: int = 100
    """Maximum items to show in workspace view (default: 100)"""

    workspace_show_private: bool = False
    """Show private (not shared) variables in workspace view (default: False)"""

    # Display settings
    display_max_preview_lines: int = 10
    """Max lines to show in data previews (default: 10)"""

    display_use_colors: bool = True
    """Use ANSI colors in terminal output (default: True)"""

    # Security settings
    auto_approve_computations: bool = False
    """Auto-approve all incoming computation requests (DANGEROUS, default: False)"""

    allow_code_execution: bool = False
    """Allow execution of remote code without approval (DANGEROUS, default: False)"""

    def __repr__(self) -> str:
        """Display current settings."""
        lines = ["━" * 70]
        lines.append("⚙️  Beaver Settings")
        lines.append("━" * 70)

        # Group settings by category
        categories = {
            "📥 Inbox Retention": [
                ("inbox_max_versions", self.inbox_max_versions, "versions"),
                ("inbox_retention_hours", self.inbox_retention_hours, "hours"),
                ("inbox_auto_cleanup", self.inbox_auto_cleanup, ""),
            ],
            "📡 Live Sync": [
                ("live_default_interval", self.live_default_interval, "seconds"),
                ("live_auto_subscribe", self.live_auto_subscribe, ""),
            ],
            "🗂️  Workspace": [
                ("workspace_max_items", self.workspace_max_items, "items"),
                ("workspace_show_private", self.workspace_show_private, ""),
            ],
            "🎨 Display": [
                ("display_max_preview_lines", self.display_max_preview_lines, "lines"),
                ("display_use_colors", self.display_use_colors, ""),
            ],
            "🔒 Security": [
                ("auto_approve_computations", self.auto_approve_computations, ""),
                ("allow_code_execution", self.allow_code_execution, ""),
            ],
        }

        for category, settings in categories.items():
            lines.append(f"\n{category}")
            for name, value, unit in settings:
                value_str = f"{value} {unit}".strip() if unit else str(value)
                lines.append(f"  {name}: {value_str}")

        lines.append("\n" + "━" * 70)
        lines.append("💡 Modify: bv.settings.inbox_max_versions = 20")
        lines.append("━" * 70)

        return "\n".join(lines)

    def reset(self):
        """Reset all settings to defaults."""
        defaults = BeaverSettings()
        for key in self.__dataclass_fields__:
            setattr(self, key, getattr(defaults, key))
        print("✓ Settings reset to defaults")
