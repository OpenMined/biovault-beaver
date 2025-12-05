"""Remote variable registry for lazy data sharing."""

from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_uv_venv() -> bool:
    """Check if we're running in a uv-managed virtual environment."""
    import os
    import sys

    # Check for uv marker file in the venv
    if hasattr(sys, "prefix"):
        uv_marker = Path(sys.prefix) / "uv.lock"
        if uv_marker.exists():
            return True
        # Also check for pyvenv.cfg with uv indicator
        pyvenv_cfg = Path(sys.prefix) / "pyvenv.cfg"
        if pyvenv_cfg.exists():
            with contextlib.suppress(Exception):
                content = pyvenv_cfg.read_text()
                if "uv" in content.lower():
                    return True

    # Check if UV_VIRTUAL_ENV is set
    if os.environ.get("UV_VIRTUAL_ENV"):
        return True

    # Check if uv command is available and we're in a venv it recognizes
    if os.environ.get("VIRTUAL_ENV"):
        import shutil

        if shutil.which("uv"):
            return True

    return False


def _prompt_install_deps(missing: list[str], var_name: str) -> bool:
    """
    Prompt user to install missing dependencies.

    Returns True if installation succeeded, False otherwise.
    """
    import shutil
    import subprocess
    import sys

    deps_str = " ".join(missing)
    use_uv = _is_uv_venv() and shutil.which("uv")
    pip_cmd = "uv pip install" if use_uv else "pip install"

    print(f"\n⚠️  Missing dependencies for '{var_name}': {', '.join(missing)}")
    print(f"   To load this data, you need to install: {deps_str}")
    print()

    try:
        response = input(f"Install now with '{pip_cmd} {deps_str}'? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        print("💡 To install manually, run:")
        print(f"   {pip_cmd} {deps_str}")
        return False

    if response in ("y", "yes"):
        print(f"\n📦 Installing {deps_str}...")
        try:
            if use_uv:
                cmd = ["uv", "pip", "install"] + missing
            else:
                cmd = [sys.executable, "-m", "pip", "install"] + missing
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print(f"✅ Successfully installed: {deps_str}")
                # Re-import to make packages available
                for pkg in missing:
                    with contextlib.suppress(Exception):
                        importlib.import_module(pkg)
                return True
            else:
                print("❌ Installation failed:")
                print(result.stderr)
                print("\n💡 To install manually, run:")
                print(f"   {pip_cmd} {deps_str}")
                return False
        except Exception as e:
            print(f"❌ Installation failed: {e}")
            print("\n💡 To install manually, run:")
            print(f"   {pip_cmd} {deps_str}")
            return False
    else:
        print("\n💡 To install manually, run:")
        print(f"   {pip_cmd} {deps_str}")
        return False


def _detect_dependencies(obj: Any) -> Dict[str, str]:
    """
    Best-effort dependency detection for shared objects.

    Captures key package versions so receivers know what to install to load.
    """
    deps: Dict[str, str] = {}

    def add(pkg: str):
        try:
            deps[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            deps[pkg] = "unknown"

    module = getattr(obj, "__module__", "") or ""
    typename = type(obj).__name__

    # AnnData
    if module.startswith("anndata") or typename == "AnnData":
        add("anndata")
        # scanpy is typically used alongside AnnData
        add("scanpy")
    # Pandas
    if module.startswith("pandas") or typename in {"DataFrame", "Series"}:
        add("pandas")
    # NumPy
    if module.startswith("numpy") or typename == "ndarray":
        add("numpy")

    return deps


class _SafeDisplayProxy:
    """
    Lightweight proxy that preserves the underlying object but guards repr/html.

    Useful for objects like AnnData whose repr can be heavy or fragile.
    Access the real object via ._raw if needed.
    """

    def __init__(self, obj: Any, label: Optional[str] = None):
        self._raw = obj
        self._label = label or type(obj).__name__

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)

    def __getitem__(self, key):
        return self._raw[key]

    def __iter__(self):
        return iter(self._raw)

    def __len__(self):
        with contextlib.suppress(Exception):
            return len(self._raw)
        raise TypeError(f"{self._label} has no len()")

    def __call__(self, *args, **kwargs):
        return self._raw(*args, **kwargs)

    def __repr__(self) -> str:
        try:
            # Lazy import to avoid cycles
            from .twin import _safe_preview
        except Exception:

            def _safe_preview(v):
                return repr(v)

        preview = _safe_preview(self._raw)
        return f"<{self._label} (preview): {preview} — real object at ._raw>"

    def _repr_html_(self) -> str:
        try:
            from .twin import _safe_preview

            preview = _safe_preview(self._raw)
        except Exception:
            preview = repr(self._raw)
        return (
            f"<b>{self._label}</b> (preview)<br>"
            f"<code>{preview}</code><br>"
            f"<i>Real object available at ._raw</i>"
        )


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
    deps: Dict[str, str] = field(default_factory=dict)  # dependency versions for loaders

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
    Can be used standalone or within a session context.
    """

    def __init__(
        self,
        owner: str,
        registry_path: Path,
        session_id: Optional[str] = None,
        context=None,
        peer: Optional[str] = None,
    ):
        """
        Initialize registry.

        Args:
            owner: Username who owns these vars
            registry_path: Path to registry JSON file
            session_id: Optional session ID if within a session
            context: Optional BeaverContext for encryption support
            peer: Optional peer email for session-scoped encryption
        """
        self.owner = owner
        self.registry_path = Path(registry_path)
        self.session_id = session_id
        self._context = context
        self.peer = peer  # For session encryption
        self.vars: Dict[str, RemoteVar] = {}
        self._load()

    def _load(self):
        """Load registry from disk (decrypting if needed)."""
        if self.registry_path.exists():
            try:
                # Use SyftBox storage to decrypt if available
                if self._context and hasattr(self._context, "_backend") and self._context._backend:
                    backend = self._context._backend
                    if backend.uses_crypto:
                        raw = bytes(backend.storage.read_with_shadow(str(self.registry_path)))
                        data = json.loads(raw.decode("utf-8"))
                    else:
                        data = json.loads(self.registry_path.read_text())
                else:
                    data = json.loads(self.registry_path.read_text())
                self.vars = {name: RemoteVar.from_dict(var_data) for name, var_data in data.items()}
            except Exception as e:
                print(f"Warning: Could not load registry from {self.registry_path}: {e}")
                self.vars = {}

    def _save(self):
        """Save registry to disk (encrypting if SyftBox backend available)."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        data = {name: var.to_dict() for name, var in self.vars.items()}
        content = json.dumps(data, indent=2)

        # Use SyftBox storage to encrypt if available
        if self._context and hasattr(self._context, "_backend") and self._context._backend:
            backend = self._context._backend
            if backend.uses_crypto and self.peer:
                # Encrypt for the peer who will read this registry
                recipients = [self.peer]
                backend.storage.write_with_shadow(
                    str(self.registry_path),
                    content.encode("utf-8"),
                    recipients=recipients,
                    hint="remote-vars-registry",
                )
                return

        # Fallback to plaintext
        self.registry_path.write_text(content)

    def add(self, obj: Any, name: Optional[str] = None) -> RemoteVar:
        """
        Publish a variable to the public registry.

        For Twin objects, publishes the public side to a shared location
        where others can load it without requiring explicit send().

        For simple types, stores the value in the registry.

        Args:
            obj: The Python object to publish
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

        data_location = None

        if is_twin:
            # For Twin, publish the public side to shared location
            summary = _summarize(obj.public)
            # Prefer the Twin's own var_type (captures underlying data type)
            var_type = getattr(obj, "var_type", f"Twin[{summary.get('type', 'unknown')}]")
            deps = _detect_dependencies(obj.public)

            # Write public Twin to shared data directory
            from .runtime import pack, write_envelope

            # Create a public-only Twin with live status preserved
            live_enabled = hasattr(obj, "_live_enabled") and obj._live_enabled
            live_interval = obj._live_interval if live_enabled else 2.0

            public_twin = Twin.public_only(
                public=obj.public,
                owner=self.owner,
                name=name,
                twin_id=obj.twin_id,
                public_id=obj.public_id,
                live_enabled=live_enabled,
                live_interval=live_interval,
            )

            # Preserve runtime live status
            if live_enabled:
                public_twin._live_enabled = True
                public_twin._live_mutable = obj._live_mutable
                public_twin._live_interval = live_interval
                public_twin._last_sync = getattr(obj, "_last_sync", None)

            public_data_dir = self.registry_path.parent / "data"
            public_data_dir.mkdir(parents=True, exist_ok=True)

            # Determine backend and recipients for encryption
            # In SyftBox, data is encrypted FOR THE RECIPIENT (peer) only
            backend = None
            recipients = None
            if self._context and hasattr(self._context, "_backend") and self._context._backend:
                backend = self._context._backend
                if backend.uses_crypto and self.peer:
                    # Encrypt for the peer who will read this data
                    recipients = [self.peer]

            # Pack and write to public directory
            env = pack(
                public_twin,
                sender=self.owner,
                name=name,
                artifact_dir=public_data_dir,
                backend=backend,
                recipients=recipients,
            )

            # Write to public data directory: shared/public/{owner}/data/{name}.beaver
            # Use encrypted storage if available
            if self._context and hasattr(self._context, "_backend") and self._context._backend:
                backend = self._context._backend
                if backend.uses_crypto:
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
                    data_path = public_data_dir / env.filename()
                    # Encrypt for the peer who will read this data
                    if self.peer:
                        recipients = [self.peer]
                        backend.storage.write_with_shadow(
                            str(data_path),
                            content,
                            recipients=recipients,
                            hint="beaver-envelope",
                        )
                        data_location = str(data_path)
                    else:
                        # No peer, write plaintext
                        data_path = write_envelope(env, out_dir=public_data_dir)
                        data_location = str(data_path)
                else:
                    data_path = write_envelope(env, out_dir=public_data_dir)
                    data_location = str(data_path)
            else:
                data_path = write_envelope(env, out_dir=public_data_dir)
                data_location = str(data_path)

            stored_value = None  # Don't keep in memory

        else:
            summary = _summarize(obj)
            # Store value for simple types
            simple_types = (str, int, float, bool, type(None))
            stored_value = obj if isinstance(obj, simple_types) else None
            var_type = summary.get("type", "unknown")
            deps = _detect_dependencies(obj)

        remote_var = RemoteVar(
            name=name,
            owner=self.owner,
            var_type=var_type,
            size_bytes=summary.get("size_bytes"),
            shape=summary.get("shape"),
            dtype=summary.get("dtype"),
            envelope_type=summary.get("envelope_type", "unknown"),
            data_location=data_location,
            _stored_value=stored_value,
            deps=deps,
        )

        self.vars[name] = remote_var
        self._save()

        if is_twin:
            print(f"📢 Published Twin '{name}' (public side available at: {data_location})")
            # Remember the Twin instance so remote computations can resolve to the owner's copy.
            remote_var._twin_reference = obj
            # If live sync is enabled, also keep registry metadata for auto-updates.
            if hasattr(obj, "_live_enabled") and obj._live_enabled:
                # Store this registry on the Twin so it can call update()
                obj._published_registry = self
                obj._published_name = name
        elif stored_value is not None:
            print(f"📢 Published '{name}' = {stored_value}")

        return remote_var

    def update(self, name: str):
        """
        Re-publish a Twin (e.g., after live sync detects changes).

        Args:
            name: Variable name to update
        """
        if name not in self.vars:
            return

        remote_var = self.vars[name]

        # Only update if we have a Twin reference
        if not hasattr(remote_var, "_twin_reference"):
            return

        twin = remote_var._twin_reference
        if twin is None:
            return

        from .runtime import pack, write_envelope
        from .twin import Twin

        # Create updated public-only Twin with live status preserved
        live_enabled = hasattr(twin, "_live_enabled") and twin._live_enabled
        live_interval = twin._live_interval if live_enabled else 2.0

        public_twin = Twin.public_only(
            public=twin.public,
            owner=self.owner,
            name=name,
            twin_id=twin.twin_id,
            public_id=twin.public_id,
            live_enabled=live_enabled,
            live_interval=live_interval,
        )

        # Preserve runtime live status
        if live_enabled:
            public_twin._live_enabled = True
            public_twin._live_mutable = twin._live_mutable
            public_twin._live_interval = live_interval
            public_twin._last_sync = getattr(twin, "_last_sync", None)

        # Write to same location (overwrite)
        public_data_dir = self.registry_path.parent / "data"
        public_data_dir.mkdir(parents=True, exist_ok=True)

        # Determine backend and recipients for encryption
        # In SyftBox, data is encrypted FOR THE RECIPIENT (peer) only
        backend = None
        recipients = None
        if self._context and hasattr(self._context, "_backend") and self._context._backend:
            backend = self._context._backend
            if backend.uses_crypto and self.peer:
                # Encrypt for the peer who will read this data
                recipients = [self.peer]

        # Pack and write to public directory
        env = pack(
            public_twin,
            sender=self.owner,
            name=name,
            artifact_dir=public_data_dir,
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
            data_path = public_data_dir / env.filename()
            backend.storage.write_with_shadow(
                str(data_path),
                content,
                recipients=recipients,
                hint="beaver-envelope",
            )
        else:
            data_path = write_envelope(env, out_dir=public_data_dir)

        # Update metadata
        remote_var.data_location = str(data_path)
        remote_var.updated_at = _iso_now()

        self._save()

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
            print("💡 Call .clear(confirm=True) to proceed")
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
        session=None,  # Session reference for session-scoped views
    ):
        """
        Initialize remote var view.

        Args:
            remote_user: The user whose vars we're viewing
            local_user: Our username
            registry_path: Path to their registry file
            data_dir: Directory where data is stored
            context: BeaverContext for loading data
            session: Session reference (if viewing within a session context)
        """
        self.remote_user = remote_user
        self.local_user = local_user
        self.registry_path = Path(registry_path)
        self.data_dir = Path(data_dir)
        self.context = context
        self.session = session  # Store session reference
        self.vars: Dict[str, RemoteVar] = {}
        self.refresh()

    def refresh(self):
        """Refresh the view by reloading the remote registry (decrypting if needed)."""
        if self.registry_path.exists():
            try:
                # Use SyftBox storage to decrypt if available
                if self.context and hasattr(self.context, "_backend") and self.context._backend:
                    backend = self.context._backend
                    if backend.uses_crypto:
                        raw = bytes(backend.storage.read_with_shadow(str(self.registry_path)))
                        data = json.loads(raw.decode("utf-8"))
                    else:
                        data = json.loads(self.registry_path.read_text())
                else:
                    data = json.loads(self.registry_path.read_text())
                self.vars = {name: RemoteVar.from_dict(var_data) for name, var_data in data.items()}
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

    def __contains__(self, name: str) -> bool:
        """Check if a var exists in this view."""
        return name in self.vars

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

    For Twin types, accessing attributes like .public will automatically
    load the latest version from the inbox.
    """

    remote_var: RemoteVar
    view: RemoteVarView
    _loaded_twin: Optional[Any] = field(default=None, init=False, repr=False)
    _last_error: Optional[Exception] = field(default=None, init=False, repr=False)

    def _missing_deps(self) -> list[str]:
        """
        Compare declared dependencies against locally installed packages.
        """
        missing = []
        for pkg, _ver in (self.remote_var.deps or {}).items():
            try:
                importlib.import_module(pkg)
            except Exception:
                missing.append(pkg)
        return missing

    def _auto_load_twin(self, *, auto_accept: bool = False):
        """Auto-load Twin from published location if this is a Twin type.

        Args:
            auto_accept: If True, automatically accept trusted loaders without prompting
        """
        # Check if already loaded
        if self._loaded_twin is not None:
            return self._loaded_twin
        # Don't retry if we already failed - prevents infinite prompt loops
        if self._last_error is not None:
            return None

        # Only auto-load for Twin types
        if not self.remote_var.var_type.startswith("Twin["):
            return None

        from .twin import Twin

        # If the RemoteVar already holds a Twin value, use it directly
        try:
            from .twin import Twin

            if isinstance(self.remote_var._stored_value, Twin):  # type: ignore[attr-defined]
                self._loaded_twin = self.remote_var._stored_value  # type: ignore[attr-defined]
                return self._loaded_twin
        except Exception:
            pass

        # Try to load from published data location
        if self.remote_var.data_location:
            try:
                import re

                from .runtime import read_envelope

                # The stored data_location is absolute from owner's perspective
                # We need to resolve it relative to our local view
                stored_path = Path(self.remote_var.data_location)
                filename = stored_path.name

                # Primary: use our local view's data directory
                if hasattr(self.view, "data_dir"):
                    data_path = Path(self.view.data_dir) / filename
                elif hasattr(self.view, "registry_path"):
                    data_path = Path(self.view.registry_path).parent / "data" / filename
                else:
                    data_path = stored_path

                # Fallback: translate stored path to our local view
                # Pattern: sandbox/<owner>/datasites/<datasite>/... -> sandbox/<us>/datasites/<datasite>/...
                if not data_path.exists():
                    stored_str = str(stored_path)
                    # Match sandbox/<identity>/datasites/ pattern
                    match = re.search(r"/sandbox/([^/]+)/datasites/", stored_str)
                    if match and hasattr(self.view, "context") and self.view.context:
                        ctx = self.view.context
                        if hasattr(ctx, "_backend") and ctx._backend:
                            # Get our local datasites root
                            our_datasites = str(ctx._backend.data_dir / "datasites")
                            # Find where /datasites/ starts in stored path
                            idx = stored_str.find("/datasites/")
                            if idx >= 0:
                                # Take everything after /datasites/ and append to our root
                                relative = stored_str[idx + len("/datasites/") :]
                                translated = Path(our_datasites) / relative
                                if translated.exists():
                                    data_path = translated

                # Try by name if still not found
                if not data_path.exists() and hasattr(self.view, "data_dir"):
                    by_name = Path(self.view.data_dir) / f"{self.remote_var.name}.beaver"
                    if by_name.exists():
                        data_path = by_name

                # If dependencies are declared, check availability before attempting load
                if self.remote_var.deps:
                    missing = self._missing_deps()
                    if missing:
                        # Prompt user to install missing dependencies
                        if _prompt_install_deps(missing, self.remote_var.name):
                            # Re-check after installation
                            missing = self._missing_deps()
                        if missing:
                            self._last_error = (
                                f"Missing dependencies: {', '.join(missing)} "
                                f"required for '{self.remote_var.name}'"
                            )
                            return None

                if data_path.exists():
                    # Use SyftBox storage to decrypt if available
                    backend = None
                    if hasattr(self.view, "context") and self.view.context:
                        ctx = self.view.context
                        if hasattr(ctx, "_backend") and ctx._backend:
                            backend = ctx._backend
                            if backend.uses_crypto:
                                # Read and decrypt using shadow pattern
                                raw = bytes(backend.storage.read_with_shadow(str(data_path)))
                                import base64
                                import json

                                from .envelope import BeaverEnvelope

                                record = json.loads(raw.decode("utf-8"))
                                payload = base64.b64decode(record["payload_b64"])
                                env = BeaverEnvelope(
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
                            else:
                                env = read_envelope(data_path)
                        else:
                            env = read_envelope(data_path)
                    else:
                        env = read_envelope(data_path)
                    twin = env.load(inject=False, auto_accept=auto_accept, backend=backend)
                    if hasattr(twin, "public"):
                        twin.public = _ensure_sparse_shapes(twin.public)
                        # Swap public for a safe display proxy, keep raw on the side
                        try:
                            import anndata  # type: ignore

                            if isinstance(twin.public, anndata.AnnData):
                                twin._public_raw = twin.public
                                twin.public = _SafeDisplayProxy(twin.public, label="AnnData")
                        except Exception:
                            pass
                    if isinstance(twin, Twin):
                        # Set source path for live sync reloading
                        twin._source_path = str(data_path)

                        # Attach session reference if loading from a session context
                        if hasattr(self.view, "session") and self.view.session is not None:
                            twin._session = self.view.session

                        # Auto-enable live sync if owner has it enabled
                        if twin.live_enabled and not getattr(twin, "_live_enabled", False):
                            twin.enable_live(interval=twin.live_interval)

                        self._loaded_twin = twin
                        return twin
            except Exception as exc:
                # Only set _last_error for permanent errors, not transient sync errors
                # Transient errors: file not found (sync in progress)
                # Permanent errors: missing dependencies, decryption failures
                exc_str = str(exc).lower()
                is_transient = (
                    "not found" in exc_str
                    or "wait for sync" in exc_str
                    or "may need to wait" in exc_str
                )
                if not is_transient:
                    self._last_error = exc
                # Fall through to inbox check
                pass

        # Fallback: check inbox for sent Twins
        if not hasattr(self.view, "context"):
            return None

        # Find the latest Twin with matching name from the remote user
        latest_twin = None
        for envelope in self.view.context.inbox():
            if envelope.name == self.remote_var.name and envelope.sender == self.remote_var.owner:
                # Load this Twin
                twin = envelope.load(inject=False)
                if isinstance(twin, Twin):
                    latest_twin = twin
                    # Keep looking for even newer ones

        if latest_twin is not None:
            # Attach session reference if loading from a session context
            if hasattr(self.view, "session") and self.view.session is not None:
                latest_twin._session = self.view.session
            self._loaded_twin = latest_twin
            return latest_twin

        return None

    def __getattr__(self, name: str):
        """
        Auto-load Twin attributes like .public, .private, .value, etc.

        This makes bv.remote.bob.counter.public work automatically.
        """
        # Try to auto-load the Twin
        twin = self._auto_load_twin()

        if twin is not None and hasattr(twin, name):
            value = getattr(twin, name)
            # Wrap AnnData-like objects to avoid heavy/fragile repr crashes
            try:
                import anndata  # type: ignore

                if isinstance(value, anndata.AnnData):
                    value = _SafeDisplayProxy(value, label="AnnData")
            except Exception:
                # Best effort — if anndata not installed, just return the value
                pass
            return value

        # If the Twin couldn't be loaded yet, but the caller is asking for .public/.private/.value,
        # try one more time and give a helpful error (including dependency hints).
        if name in {"public", "private", "value"}:
            twin = self._auto_load_twin()
            if twin is not None and hasattr(twin, name):
                value = getattr(twin, name)
                try:
                    import anndata  # type: ignore

                    if isinstance(value, anndata.AnnData):
                        value = _SafeDisplayProxy(value, label="AnnData")
                except Exception:
                    pass
                return value
            # Build a helpful error message
            if self.remote_var.deps:
                missing = self._missing_deps()
                if missing:
                    import shutil

                    deps_str = " ".join(missing)
                    use_uv = _is_uv_venv() and shutil.which("uv")
                    pip_cmd = "uv pip install" if use_uv else "pip install"
                    msg = (
                        f"\n\n❌ Cannot access '{self.remote_var.name}.{name}' - missing dependencies\n\n"
                        f"   Required packages not installed: {', '.join(missing)}\n\n"
                        f"   To fix, run:\n"
                        f"   {pip_cmd} {deps_str}\n\n"
                        f"   Then retry: {self.remote_var.name}.{name}\n"
                    )
                    raise AttributeError(msg)
            msg = "Remote Twin data not available."
            if getattr(self, "_last_error", None):
                msg += f" Load error: {self._last_error}"
            else:
                msg += f" Ask the owner to publish/send '{self.remote_var.name}'."
            raise AttributeError(msg)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def load(
        self, *, inject: bool = True, as_name: Optional[str] = None, auto_accept: bool = False
    ):
        """
        Load the remote variable data.

        For Twin types, requests and loads the public side from the owner.
        For simple types, returns the stored value if available.

        Args:
            inject: Whether to inject into globals
            as_name: Name to use in globals (defaults to remote var name)
            auto_accept: If True, automatically accept trusted loaders without prompting

        Returns:
            The loaded object (Twin, simple value, or pointer if not available)
        """
        var_name = as_name or self.remote_var.name

        # Check if this is a Twin type
        is_twin = self.remote_var.var_type.startswith("Twin[")

        if is_twin:
            # Try to load from published data location first
            from .twin import Twin

            twin = self._auto_load_twin(auto_accept=auto_accept)
            if twin is not None:
                if inject:
                    import inspect

                    frame = inspect.currentframe()
                    if frame and frame.f_back:
                        caller_globals = frame.f_back.f_globals
                        caller_globals[var_name] = twin
                        print(f"✓ Loaded Twin '{var_name}' from published location")
                return twin

            # Fallback: Check if we already have the Twin in the context's remote_vars
            # (it would have been sent when they published it)
            if hasattr(self.view, "context") and hasattr(self.view.context, "remote_vars"):
                for var in self.view.context.remote_vars.vars.values():
                    if (
                        var.var_id == self.remote_var.var_id
                        and var._stored_value is not None
                        and isinstance(var._stored_value, Twin)
                    ):
                        twin = var._stored_value
                        if inject:
                            import inspect

                            frame = inspect.currentframe()
                            if frame and frame.f_back:
                                caller_globals = frame.f_back.f_globals
                                caller_globals[var_name] = twin
                                print(f"✓ Loaded Twin '{var_name}' into globals")
                        return twin

            # Twin not found locally - need to request it or wait for sync
            # Check if this is a transient error (sync in progress) vs permanent error
            last_error = getattr(self, "_last_error", None)
            is_transient = last_error is None  # No permanent error stored = transient (sync)

            if self.remote_var.data_location:
                if last_error:
                    print(
                        f"⚠️  Twin '{var_name}' published at {self.remote_var.data_location} "
                        f"but could not be loaded: {last_error}"
                    )
                    print("💡 Install required dependencies (e.g., anndata/scanpy) and retry.")
                else:
                    # Transient - likely waiting for sync, don't print much
                    pass
            else:
                print(f"📍 Twin '{var_name}' metadata found, but data not yet sent")
                print("💡 The owner needs to explicitly .send() this Twin")
                print(
                    f"💡 Or you can request access with: bv.peer('{self.remote_var.owner}').send_request('{var_name}')"
                )

            # For transient errors (sync in progress), return None so retry loops continue
            # For permanent errors, inject pointer and return self
            if is_transient:
                return None

            if inject:
                import inspect

                frame = inspect.currentframe()
                if frame and frame.f_back:
                    caller_globals = frame.f_back.f_globals
                    # Inject the pointer for now
                    caller_globals[var_name] = self
                    print(
                        f"✓ Injected pointer '{var_name}' into globals (will need actual data to use)"
                    )

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

            print("📍 Pointer only - no data loaded yet")
            print(f"💡 Request data from {self.remote_var.owner}")

            return self

    def __repr__(self) -> str:
        # Don't retry loading if we already have an error - avoids infinite prompt loops
        if self._loaded_twin is not None:
            return repr(self._loaded_twin)
        if self._last_error is not None:
            return repr(self.remote_var)
        twin = self._auto_load_twin()
        if twin is not None:
            return repr(twin)
        return repr(self.remote_var)

    def _repr_html_(self) -> str:
        # Don't retry loading if we already have an error - avoids infinite prompt loops
        if self._loaded_twin is not None:
            twin = self._loaded_twin
        elif self._last_error is not None:
            twin = None
        else:
            twin = self._auto_load_twin()
        if twin is not None and hasattr(twin, "_repr_html_"):
            return twin._repr_html_()  # type: ignore[attr-defined]
        return (
            self.remote_var._repr_html_() if hasattr(self.remote_var, "_repr_html_") else repr(self)
        )


def _ensure_sparse_shapes(val: Any) -> Any:
    """
    Ensure scipy sparse matrices keep their shape metadata after deserialization.
    """
    try:
        import scipy.sparse as sp
    except Exception:
        return val

    def fix(obj):
        if sp.issparse(obj) and not hasattr(obj, "_shape"):
            obj._shape = obj.shape  # type: ignore[attr-defined]
        return obj

    # Top-level
    fix(val)

    # AnnData specific: fix X, layers, obsm/varm, obsp/varp
    try:
        import anndata  # type: ignore

        if isinstance(val, anndata.AnnData):
            fix(val.X)
            for container in (val.layers, val.obsm, val.varm, val.obsp, val.varp):
                for k in list(container.keys()):
                    container[k] = fix(container[k])
    except Exception:
        pass

    return val
