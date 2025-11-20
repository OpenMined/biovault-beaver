"""LiveMixin: Real-time synchronization capability for RemoteData."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional


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
        if hasattr(super(), '__post_init__'):
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
        return getattr(self, '_live_enabled', False)

    @property
    def mutable(self) -> bool:
        """Check if live sync is mutable."""
        return getattr(self, '_live_mutable', False)

    @property
    def sync_interval(self) -> float:
        """Get sync interval in seconds."""
        return getattr(self, '_live_interval', 2.0)

    @property
    def last_sync(self) -> Optional[str]:
        """Get timestamp of last sync."""
        return getattr(self, '_last_sync', None)

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
        if not hasattr(self, '_live_enabled'):
            self._live_enabled = False
            self._live_mutable = False
            self._live_interval = 2.0
            self._live_thread = None
            self._live_stop = False
            self._last_sync = None
            self._last_value_hash = None
            self._subscribers = []

        if self._live_enabled:
            print(f"⚠️  Live sync already enabled")
            return

        self._live_enabled = True
        self._live_mutable = mutable
        self._live_interval = interval
        self._live_stop = False

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
        if not hasattr(self, '_live_enabled'):
            return
        if not self._live_enabled:
            return

        self._live_enabled = False
        self._live_stop = True

        # Wait for thread to stop
        if self._live_thread and self._live_thread.is_alive():
            self._live_thread.join(timeout=self._live_interval + 1)

        self._live_thread = None
        print(f"⚫ Live sync disabled")

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
                if hasattr(self, 'get_value'):
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
                value_lines = value.split('\n')
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
        if not hasattr(self, 'get_value'):
            return

        try:
            # Get current value without triggering print messages
            # For Twin objects, prefer public side for change detection since
            # that's what gets sent to subscribers
            if hasattr(self, 'public') and self.public is not None:
                current_value = self.public
            elif hasattr(self, 'private') and self.private is not None:
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
            changed = (self._last_value_hash is not None and
                      current_hash != self._last_value_hash)

            # Debug output
            if hasattr(self, 'name'):
                actual_public = getattr(self, 'public', 'N/A')
                print(f"[LiveSync Debug] Twin '{self.name}': value={current_value}, actual_public={actual_public}, hash={current_hash}, last_hash={self._last_value_hash}, changed={changed}")

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

                # Send updates to remote subscribers (Twin only)
                if hasattr(self, '_live_subscribers') and hasattr(self, '_live_context'):
                    if self._live_subscribers and self._live_context:
                        self._send_live_update()

        except Exception as e:
            # Don't crash the sync thread
            pass

    def _send_live_update(self):
        """Send Twin update to all live subscribers."""
        if not hasattr(self, '_live_subscribers') or not hasattr(self, '_live_context'):
            return

        if not self._live_subscribers or not self._live_context:
            return

        from .twin import Twin
        if not isinstance(self, Twin):
            return

        # Send updated Twin to each subscriber
        for subscriber_user in self._live_subscribers:
            try:
                result = self._live_context.send(
                    self,
                    user=subscriber_user,
                    name=self.name or f"twin_{self.twin_id[:8]}"
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

        # Handle case where LiveMixin wasn't initialized (deserialized objects)
        if not hasattr(self, '_live_enabled'):
            lines.append(f"  Live: ⚫ Disabled")
            return lines

        if self._live_enabled:
            mode = "mutable" if self._live_mutable else "read-only"
            lines.append(f"  Live: 🟢 Enabled ({mode}, {self._live_interval}s)")
            if self._last_sync:
                lines.append(f"  Last sync: {self._last_sync}")
        else:
            lines.append(f"  Live: ⚫ Disabled")

        return lines
