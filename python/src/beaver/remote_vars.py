"""Remote variable registry for lazy data sharing."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RemoteVar:
    """
    Metadata pointer to a remote variable.

    The actual data is not stored here - this is just a catalog entry
    that allows discovery and on-demand loading.
    """

    var_id: str = field(default_factory=lambda: uuid4().hex)
    name: str = ""
    owner: str = "unknown"
    var_type: str = "unknown"
    size_bytes: Optional[int] = None
    shape: Optional[str] = None
    dtype: Optional[str] = None
    envelope_type: str = "unknown"  # "code" or "data"
    created_at: str = field(default_factory=_iso_now)
    updated_at: str = field(default_factory=_iso_now)
    data_location: Optional[str] = None  # Path to actual .beaver file if sent
    _stored_value: Optional[Any] = None  # Actual value for simple types

    def __repr__(self) -> str:
        type_info = self.var_type
        if self.shape:
            type_info += f" {self.shape}"
        if self.dtype:
            type_info += f" ({self.dtype})"

        status = "📍 pointer" if not self.data_location else "✓ loaded"

        return (
            f"RemoteVar({self.name!r})\n"
            f"  ID: {self.var_id[:12]}...\n"
            f"  Owner: {self.owner}\n"
            f"  Type: {type_info}\n"
            f"  Status: {status}"
        )

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> RemoteVar:
        """Create from dict."""
        return cls(**data)


class RemoteVarRegistry:
    """
    Registry of published remote variables for a user.

    Manages the catalog of variables that are available for others to load.
    """

    def __init__(self, owner: str, registry_path: Path):
        """
        Initialize registry.

        Args:
            owner: Username who owns these vars
            registry_path: Path to registry JSON file
        """
        self.owner = owner
        self.registry_path = Path(registry_path)
        self.vars: Dict[str, RemoteVar] = {}
        self._load()

    def _load(self):
        """Load registry from disk."""
        if self.registry_path.exists():
            try:
                data = json.loads(self.registry_path.read_text())
                self.vars = {
                    name: RemoteVar.from_dict(var_data)
                    for name, var_data in data.items()
                }
            except Exception as e:
                print(f"Warning: Could not load registry from {self.registry_path}: {e}")
                self.vars = {}

    def _save(self):
        """Save registry to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        data = {name: var.to_dict() for name, var in self.vars.items()}
        self.registry_path.write_text(json.dumps(data, indent=2))

    def add(self, obj: Any, name: Optional[str] = None) -> RemoteVar:
        """
        Add a variable to the registry (metadata only, no data sent).

        For Twin objects, only the public side will be shared when accessed remotely.

        Args:
            obj: The Python object to register
            name: Variable name (auto-detected from caller if None)

        Returns:
            RemoteVar metadata object
        """
        # Auto-detect name if not provided
        if name is None:
            # Try to get from caller's frame
            import inspect

            frame = inspect.currentframe()
            if frame and frame.f_back:
                caller_locals = frame.f_back.f_locals
                # Find the variable name that matches this object
                for var_name, var_obj in caller_locals.items():
                    if var_obj is obj and not var_name.startswith("_"):
                        name = var_name
                        break

        if not name:
            name = "unnamed"

        # Check if this is a Twin object
        from .twin import Twin
        is_twin = isinstance(obj, Twin)

        # Create metadata
        from .runtime import _summarize

        if is_twin:
            # For Twin, summarize the public side
            summary = _summarize(obj.public)
            # Don't store Twin in registry - it will be sent separately
            # The registry is just metadata
            stored_value = None
            var_type = f"Twin[{summary.get('type', 'unknown')}]"
        else:
            summary = _summarize(obj)
            # Store value for simple types
            simple_types = (str, int, float, bool, type(None))
            stored_value = obj if isinstance(obj, simple_types) else None
            var_type = summary.get("type", "unknown")

        remote_var = RemoteVar(
            name=name,
            owner=self.owner,
            var_type=var_type,
            size_bytes=summary.get("size_bytes"),
            shape=summary.get("shape"),
            dtype=summary.get("dtype"),
            envelope_type=summary.get("envelope_type", "unknown"),
            _stored_value=stored_value,
        )

        self.vars[name] = remote_var
        self._save()

        if is_twin:
            print(f"🔒 Added Twin '{name}' (public side will be shared)")

        return remote_var

    def remove(self, name_or_id: str) -> bool:
        """
        Remove a variable from the registry by name or ID.

        Args:
            name_or_id: Variable name or var_id (full or prefix)

        Returns:
            True if removed, False if not found
        """
        # Try exact name match first
        if name_or_id in self.vars:
            del self.vars[name_or_id]
            self._save()
            print(f"🗑️  Removed: {name_or_id}")
            return True

        # Try ID match
        for name, var in self.vars.items():
            if var.var_id == name_or_id or var.var_id.startswith(name_or_id):
                del self.vars[name]
                self._save()
                print(f"🗑️  Removed: {name} (ID: {var.var_id[:12]}...)")
                return True

        print(f"⚠️  Not found: {name_or_id}")
        return False

    def remove_by_id(self, var_id: str) -> bool:
        """
        Remove a variable by its ID.

        Args:
            var_id: Variable ID (full or prefix)

        Returns:
            True if removed, False if not found
        """
        for name, var in self.vars.items():
            if var.var_id == var_id or var.var_id.startswith(var_id):
                del self.vars[name]
                self._save()
                print(f"🗑️  Removed: {name} (ID: {var.var_id[:12]}...)")
                return True
        print(f"⚠️  ID not found: {var_id}")
        return False

    def remove_many(self, *names_or_ids: str) -> int:
        """
        Remove multiple variables at once.

        Args:
            *names_or_ids: Variable names or IDs

        Returns:
            Number of variables removed
        """
        count = 0
        for name_or_id in names_or_ids:
            if self.remove(name_or_id):
                count += 1
        return count

    def clear(self, *, confirm: bool = False):
        """
        Remove all variables from the registry.

        Args:
            confirm: Must be True to actually clear (safety check)
        """
        if not confirm:
            count = len(self.vars)
            print(f"⚠️  This will remove {count} variable(s)")
            print(f"💡 Call .clear(confirm=True) to proceed")
            return

        count = len(self.vars)
        self.vars.clear()
        self._save()
        print(f"🗑️  Cleared {count} variable(s)")

    def get(self, name: str) -> Optional[RemoteVar]:
        """Get a remote var by name."""
        return self.vars.get(name)

    def __setitem__(self, name: str, obj: Any):
        """Allow registry['name'] = obj syntax."""
        self.add(obj, name=name)

    def __getitem__(self, name: str) -> RemoteVar:
        """Allow registry['name'] syntax."""
        if name not in self.vars:
            raise KeyError(f"Remote var not found: {name}")
        return self.vars[name]

    def __delitem__(self, name: str):
        """Allow del registry['name'] syntax."""
        self.remove(name)

    def __contains__(self, name: str) -> bool:
        """Check if a var exists."""
        return name in self.vars

    def __repr__(self) -> str:
        """String representation."""
        if not self.vars:
            return f"RemoteVarRegistry({self.owner}): empty"

        lines = [f"RemoteVarRegistry({self.owner}):"]
        for name, var in self.vars.items():
            type_info = var.var_type
            if var.shape:
                type_info += f" {var.shape}"
            status = "📍" if not var.data_location else "✓"
            lines.append(f"  {status} {name}: {type_info}")

        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter."""
        if not self.vars:
            return f"<b>RemoteVarRegistry({self.owner})</b>: empty"

        rows = []
        for name, var in self.vars.items():
            type_info = var.var_type
            if var.shape:
                type_info += f" {var.shape}"

            status = "📍 pointer" if not var.data_location else "✓ loaded"

            rows.append(
                f"<tr>"
                f"<td><b>{name}</b></td>"
                f"<td>{type_info}</td>"
                f"<td>{status}</td>"
                f"<td><code>{var.var_id[:12]}...</code></td>"
                f"</tr>"
            )

        return (
            f"<b>RemoteVarRegistry({self.owner})</b><br>"
            f"<table>"
            f"<thead><tr><th>Name</th><th>Type</th><th>Status</th><th>ID</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            f"</table>"
        )


class RemoteVarView:
    """
    View of another user's remote variables.

    Allows browsing and loading remote vars from another user.
    """

    def __init__(
        self,
        remote_user: str,
        local_user: str,
        registry_path: Path,
        data_dir: Path,
        context,
    ):
        """
        Initialize remote var view.

        Args:
            remote_user: The user whose vars we're viewing
            local_user: Our username
            registry_path: Path to their registry file
            data_dir: Directory where data is stored
            context: BeaverContext for loading data
        """
        self.remote_user = remote_user
        self.local_user = local_user
        self.registry_path = Path(registry_path)
        self.data_dir = Path(data_dir)
        self.context = context
        self.vars: Dict[str, RemoteVar] = {}
        self.refresh()

    def refresh(self):
        """Refresh the view by reloading the remote registry."""
        if self.registry_path.exists():
            try:
                data = json.loads(self.registry_path.read_text())
                self.vars = {
                    name: RemoteVar.from_dict(var_data)
                    for name, var_data in data.items()
                }
            except Exception as e:
                print(f"Warning: Could not load remote registry: {e}")
                self.vars = {}

    def __getitem__(self, name: str) -> RemoteVarPointer:
        """Get a pointer to a remote var."""
        if name not in self.vars:
            raise KeyError(f"Remote var not found: {name}")

        return RemoteVarPointer(
            remote_var=self.vars[name],
            view=self,
        )

    def __repr__(self) -> str:
        """String representation."""
        if not self.vars:
            return f"RemoteVarView({self.remote_user}): empty"

        lines = [f"RemoteVarView({self.remote_user}):"]
        for name, var in self.vars.items():
            type_info = var.var_type
            if var.shape:
                type_info += f" {var.shape}"
            lines.append(f"  📡 {name}: {type_info}")

        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter."""
        if not self.vars:
            return f"<b>RemoteVarView({self.remote_user})</b>: empty"

        rows = []
        for name, var in self.vars.items():
            type_info = var.var_type
            if var.shape:
                type_info += f" {var.shape}"

            rows.append(
                f"<tr>"
                f"<td><b>{name}</b></td>"
                f"<td>{type_info}</td>"
                f"<td><code>{var.var_id[:12]}...</code></td>"
                f"</tr>"
            )

        return (
            f"<b>RemoteVarView({self.remote_user})</b><br>"
            f"<table>"
            f"<thead><tr><th>Name</th><th>Type</th><th>ID</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            f"</table>"
        )


@dataclass
class RemoteVarPointer:
    """
    Pointer to a specific remote variable that can be loaded on demand.
    """

    remote_var: RemoteVar
    view: RemoteVarView

    def load(self, *, inject: bool = True, as_name: Optional[str] = None):
        """
        Load the remote variable data.

        For Twin types, requests and loads the public side from the owner.
        For simple types, returns the stored value if available.

        Args:
            inject: Whether to inject into globals
            as_name: Name to use in globals (defaults to remote var name)

        Returns:
            The loaded object (Twin, simple value, or pointer if not available)
        """
        var_name = as_name or self.remote_var.name

        # Check if this is a Twin type
        is_twin = self.remote_var.var_type.startswith('Twin[')

        if is_twin:
            # Request the Twin from the owner
            # The owner should have already sent it if it's in their remote_vars
            from .runtime import pack, write_envelope
            from .twin import Twin

            # Check if we already have the Twin in the context's remote_vars
            # (it would have been sent when they published it)
            if hasattr(self.view, 'context') and hasattr(self.view.context, 'remote_vars'):
                for var in self.view.context.remote_vars.vars.values():
                    if var.var_id == self.remote_var.var_id and var._stored_value is not None:
                        if isinstance(var._stored_value, Twin):
                            twin = var._stored_value
                            if inject:
                                import inspect
                                frame = inspect.currentframe()
                                if frame and frame.f_back:
                                    caller_globals = frame.f_back.f_globals
                                    caller_globals[var_name] = twin
                                    print(f"✓ Loaded Twin '{var_name}' into globals")
                            return twin

            # Twin not found locally - need to request it
            print(f"📍 Twin '{var_name}' metadata found, but data not yet sent")
            print(f"💡 The owner needs to explicitly .send() this Twin")
            print(f"💡 Or you can request access with: bv.peer('{self.remote_var.owner}').send_request('{var_name}')")

            if inject:
                import inspect
                frame = inspect.currentframe()
                if frame and frame.f_back:
                    caller_globals = frame.f_back.f_globals
                    # Inject the pointer for now
                    caller_globals[var_name] = self
                    print(f"✓ Injected pointer '{var_name}' into globals (will need actual data to use)")

            return self

        # For simple types, return stored value if available
        elif self.remote_var._stored_value is not None:
            value = self.remote_var._stored_value
            if inject:
                import inspect
                frame = inspect.currentframe()
                if frame and frame.f_back:
                    caller_globals = frame.f_back.f_globals
                    caller_globals[var_name] = value
                    print(f"✓ Loaded '{var_name}' = {value}")
            return value

        # No data available
        else:
            if inject:
                import inspect
                frame = inspect.currentframe()
                if frame and frame.f_back:
                    caller_globals = frame.f_back.f_globals
                    caller_globals[var_name] = self
                    print(f"✓ Injected pointer '{var_name}' into globals")
                    print(f"💡 Use {var_name} in your code - data will load on-demand (future)")

            print(f"📍 Pointer only - no data loaded yet")
            print(f"💡 Request data from {self.remote_var.owner}")

            return self

    def __repr__(self) -> str:
        return repr(self.remote_var)
