"""LiveMixin: Real-time synchronization capability for RemoteData."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Callable, Optional


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LiveMixin:
    """
    Mixin that adds live synchronization capability to RemoteData.

    Can be enabled on any RemoteData to get real-time updates.
    Supports read-only (default) or mutable (last-write-wins) modes.
    """

    def __post_init__(self):
        """Initialize live sync state."""
        # Call parent __post_init__ if it exists
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

        # Live sync state
        self._live_enabled = False
        self._live_mutable = False
        self._live_interval = 2.0
        self._live_thread: Optional[threading.Thread] = None
        self._live_stop = False
        self._last_sync: Optional[str] = None
        self._last_value_hash: Optional[int] = None

        # Subscribers (callbacks on change)
        self._subscribers: list[Callable] = []

    @property
    def live(self) -> bool:
        """Check if live sync is enabled."""
        return getattr(self, "_live_enabled", False)

    @property
    def mutable(self) -> bool:
        """Check if live sync is mutable."""
        return getattr(self, "_live_mutable", False)

    @property
    def sync_interval(self) -> float:
        """Get sync interval in seconds."""
        return getattr(self, "_live_interval", 2.0)

    @property
    def last_sync(self) -> Optional[str]:
        """Get timestamp of last sync."""
        return getattr(self, "_last_sync", None)

    def enable_live(
        self,
        *,
        mutable: bool = False,
        interval: float = 2.0,
    ):
        """
        Enable live synchronization.

        Args:
            mutable: Allow writes (last-write-wins)
            interval: Sync interval in seconds
        """
        # Initialize attributes if not present (deserialized objects)
        if not hasattr(self, "_live_enabled"):
            self._live_enabled = False
            self._live_mutable = False
            self._live_interval = 2.0
            self._live_thread = None
            self._live_stop = False
            self._last_sync = None
            self._last_value_hash = None
            self._subscribers = []

        if self._live_enabled:
            print("⚠️  Live sync already enabled")
            return

        self._live_enabled = True
        self._live_mutable = mutable
        self._live_interval = interval
        self._live_stop = False

        # Also set dataclass fields for serialization (Twin only)
        from .twin import Twin

        if isinstance(self, Twin):
            object.__setattr__(self, "live_enabled", True)
            object.__setattr__(self, "live_interval", interval)

        # Start background sync thread
        self._live_thread = threading.Thread(
            target=self._live_sync_loop,
            daemon=True,
            name=f"live-{self.id[:8] if hasattr(self, 'id') else 'unknown'}",
        )
        self._live_thread.start()

        mode = "mutable" if mutable else "read-only"
        print(f"🟢 Live sync enabled ({mode}, every {interval}s)")

    def disable_live(self):
        """Disable live synchronization."""
        # Initialize if not present (deserialized objects)
        if not hasattr(self, "_live_enabled"):
            return
        if not self._live_enabled:
            return

        self._live_enabled = False
        self._live_stop = True

        # Also update dataclass field for serialization (Twin only)
        from .twin import Twin

        if isinstance(self, Twin):
            object.__setattr__(self, "live_enabled", False)

        # Wait for thread to stop
        if self._live_thread and self._live_thread.is_alive():
            self._live_thread.join(timeout=self._live_interval + 1)

        self._live_thread = None
        print("⚫ Live sync disabled")

    def on_change(self, callback: Callable):
        """
        Subscribe to change notifications.

        Args:
            callback: Function to call when data changes
        """
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def off_change(self, callback: Callable):
        """
        Unsubscribe from change notifications.

        Args:
            callback: Function to remove
        """
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def watch(self, *, timeout: Optional[float] = None):
        """
        Watch for changes (generator pattern).

        Yields the updated value each time it changes.

        Args:
            timeout: Stop watching after N seconds (None = forever)

        Example:
            for value in data.watch(timeout=30):
                print(f"Updated: {value}")
        """
        if not self._live_enabled:
            raise RuntimeError("Live sync not enabled - call enable_live() first")

        start_time = time.time()
        last_hash = self._last_value_hash

        while True:
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                break

            # Check for changes
            if self._last_value_hash != last_hash:
                last_hash = self._last_value_hash
                if hasattr(self, "get_value"):
                    yield self.get_value()

            time.sleep(0.1)  # Small sleep to avoid busy waiting

    def tail(self, *, lines: Optional[int] = None, timeout: Optional[float] = None):
        """
        Tail the data like `tail -f` (for log-style data).

        Args:
            lines: Show last N lines (None = all)
            timeout: Stop tailing after N seconds (None = forever)

        Example:
            for line in logs.tail():
                print(line)
        """
        for value in self.watch(timeout=timeout):
            # If value is list/string, show lines
            if isinstance(value, str):
                value_lines = value.split("\n")
                if lines:
                    value_lines = value_lines[-lines:]
                for line in value_lines:
                    if line:  # Skip empty lines
                        yield line
            else:
                # For other types, just yield the value
                yield value

    def _live_sync_loop(self):
        """Background thread that syncs data periodically."""
        while not self._live_stop:
            try:
                self._sync_once()
            except Exception as e:
                print(f"⚠️  Live sync error: {e}")

            # Sleep in small increments to be responsive to stop signal
            for _ in range(int(self._live_interval * 10)):
                if self._live_stop:
                    break
                time.sleep(0.1)

    def _sync_once(self):
        """Perform one sync operation - detects changes and sends updates."""
        if not hasattr(self, "get_value"):
            return

        try:
            # Check if this is a subscriber (has source path) or owner (has private data)
            from .twin import Twin

            is_subscriber = (
                isinstance(self, Twin)
                and hasattr(self, "_source_path")
                and self._source_path is not None
                and self.private is None
            )

            if is_subscriber:
                # SUBSCRIBER MODE: Reload from source and detect changes
                self._sync_as_subscriber()
            else:
                # OWNER MODE: Detect local changes and publish
                self._sync_as_owner()

        except Exception:
            # Don't crash the sync thread
            pass

    def _sync_as_subscriber(self):
        """Sync mode for subscribers - reload from source."""
        from .twin import Twin

        if not isinstance(self, Twin):
            return

        # Check if we have source tracking info
        if not hasattr(self, "_source_path") or not self._source_path:
            return

        try:
            import json
            from pathlib import Path

            from .runtime import read_envelope

            # First, check if the source path has changed (owner updated)
            # by reading the registry metadata
            source_path = Path(self._source_path)

            # Parse the owner and check their registry
            # Format: shared/public/{owner}/data/{file}.beaver
            parts = source_path.parts
            if "public" in parts and len(parts) >= 4:
                owner_idx = parts.index("public") + 1
                if owner_idx < len(parts):
                    registry_path = source_path.parent.parent / "remote_vars.json"

                    # Read registry to get latest data_location
                    if registry_path.exists():
                        with open(registry_path) as f:
                            registry_data = json.load(f)
                            # Registry is a dict with var names as keys
                            if self.name in registry_data:
                                var_data = registry_data[self.name]
                                if var_data.get("var_type", "").startswith("Twin["):
                                    # Update source path if it changed
                                    new_location = var_data.get("data_location")
                                    if new_location and new_location != str(source_path):
                                        source_path = Path(new_location)
                                        self._source_path = new_location

            if not source_path.exists():
                return

            # Load the updated Twin
            env = read_envelope(source_path)
            updated_twin = env.load(inject=False)

            if not isinstance(updated_twin, Twin):
                return

            # Check if value changed
            old_value = self.public
            new_value = updated_twin.public

            # Update our public value if changed
            if old_value != new_value:
                object.__setattr__(self, "public", new_value)
                self._last_sync = _iso_now()

                # Notify local callbacks
                for callback in self._subscribers:
                    try:
                        callback()
                    except Exception as e:
                        print(f"⚠️  Subscriber callback error: {e}")

        except Exception:
            # Don't crash on reload errors
            pass

    def _sync_as_owner(self):
        """Sync mode for owners - detect local changes and publish."""
        # Get current value without triggering print messages
        # For Twin objects, prefer public side for change detection since
        # that's what gets sent to subscribers
        if hasattr(self, "public") and self.public is not None:
            current_value = self.public
        elif hasattr(self, "private") and self.private is not None:
            current_value = self.private
        else:
            # Fallback to get_value for other RemoteData types
            current_value = self.get_value()

        # Compute hash to detect changes
        try:
            current_hash = hash(str(current_value))
        except TypeError:
            # For unhashable types, use id
            current_hash = id(current_value)

        # Check if changed
        changed = self._last_value_hash is not None and current_hash != self._last_value_hash

        self._last_value_hash = current_hash
        self._last_sync = _iso_now()

        # If changed, send updates to live subscribers
        if changed:
            # Notify local callbacks
            for callback in self._subscribers:
                try:
                    callback()
                except Exception as e:
                    print(f"⚠️  Subscriber callback error: {e}")

            # Re-publish to public registry if this Twin is published
            from .twin import Twin

            if (
                isinstance(self, Twin)
                and hasattr(self, "_published_registry")
                and hasattr(self, "_published_name")
                and self._published_registry
                and self._published_name
            ):
                try:
                    self._published_registry.update(self._published_name)
                    print("  📢 Re-published to public registry")
                except Exception as e:
                    print(f"  ⚠️  Failed to re-publish: {e}")

            # Send updates to remote subscribers (Twin only)
            if (
                hasattr(self, "_live_subscribers")
                and hasattr(self, "_live_context")
                and self._live_subscribers
                and self._live_context
            ):
                self._send_live_update()

    def _send_live_update(self):
        """Send Twin update to all live subscribers."""
        if not hasattr(self, "_live_subscribers") or not hasattr(self, "_live_context"):
            return

        if not self._live_subscribers or not self._live_context:
            return

        from .twin import Twin

        if not isinstance(self, Twin):
            return

        # Send updated Twin to each subscriber
        for subscriber_user in self._live_subscribers:
            try:
                self._live_context.send(
                    self, user=subscriber_user, name=self.name or f"twin_{self.twin_id[:8]}"
                )
                # Debug: confirm send
                print(f"  📤 Sent live update to {subscriber_user}")
            except Exception as e:
                # Don't crash on send errors
                print(f"  ⚠️  Failed to send to {subscriber_user}: {e}")
                pass

    def _display_live_status(self) -> list[str]:
        """Generate live status lines for display."""
        lines = []

        # Check both runtime attribute and dataclass field (for deserialized Twins)
        is_live = False
        interval = 2.0

        # First check runtime attribute (for active Twins)
        if hasattr(self, "_live_enabled") and self._live_enabled:
            is_live = True
            interval = self._live_interval
        # Fall back to dataclass field (for deserialized Twins)
        elif hasattr(self, "live_enabled") and self.live_enabled:
            is_live = True
            interval = self.live_interval

        if is_live:
            mode = "mutable" if getattr(self, "_live_mutable", False) else "read-only"
            lines.append(f"  Live: 🟢 Enabled ({mode}, {interval}s)")
            if hasattr(self, "_last_sync") and self._last_sync:
                lines.append(f"  Last sync: {self._last_sync}")
            else:
                lines.append("  💡 Owner's Twin is live-syncing")
        else:
            lines.append("  Live: ⚫ Disabled")

        return lines
