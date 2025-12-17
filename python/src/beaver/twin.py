"""Twin: Privacy-preserving dual-value objects for collaborative data science."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

from .live_mixin import LiveMixin
from .remote_data import RemoteData

# Global registry of Twins with private data (for computation resolution)
# Key: (twin_id, owner) -> Twin instance
_TWIN_REGISTRY = {}


def _sync_twin_instances(twin_id: str, owner: str, source_twin):
    """
    Sync all Twin instances with the same twin_id across globals to match source_twin.

    This ensures that all variables (e.g., 'violin_result' and 'make_violin_result')
    that reference Twins with the same twin_id point to the same canonical object.
    """
    import inspect

    # Collect all potential global namespaces to search
    namespaces_to_check = []

    # Get the caller's globals by walking up the stack
    frame = inspect.currentframe()
    if frame:
        current_frame = frame.f_back
        while current_frame:
            frame_globals = current_frame.f_globals
            # Add any globals we find (prefer __main__ or ipykernel)
            if frame_globals not in namespaces_to_check:
                namespaces_to_check.append(frame_globals)
            current_frame = current_frame.f_back

    # Also try to get IPython's user namespace if available
    try:
        import IPython

        ipython = IPython.get_ipython()
        if ipython and hasattr(ipython, "user_ns") and ipython.user_ns not in namespaces_to_check:
            namespaces_to_check.append(ipython.user_ns)
    except (ImportError, AttributeError):
        pass

    # Find all Twin instances and choose the canonical one
    all_twins = []  # List of (var_name, var_value, namespace)

    # Import here to avoid circular dependency
    try:
        from .computation import ComputationRequest as computation_request_cls  # noqa: N813
    except ImportError:
        computation_request_cls = None

    for namespace in namespaces_to_check:
        for _var_name, var_value in list(namespace.items()):
            # Skip private variables and modules
            if _var_name.startswith("_") or _var_name in ("In", "Out"):
                continue

            if isinstance(var_value, Twin):
                var_twin_id = getattr(var_value, "twin_id", None)
                var_owner = getattr(var_value, "owner", None)

                if var_twin_id == twin_id and var_owner == owner:
                    all_twins.append((_var_name, var_value, namespace))

    if not all_twins:
        return

    # Choose canonical Twin: prefer one with public data, or source_twin, or first found
    canonical = source_twin
    for _var_name, var_value, _namespace in all_twins:
        # Prefer Twin that has public data (it's the original local Twin)
        if getattr(var_value, "public", None) is not None:
            canonical = var_value
            break

    # Merge data from all Twins (including source_twin) into canonical
    twins_to_merge = [source_twin] + [t[1] for t in all_twins if t[1] is not canonical]

    for twin in twins_to_merge:
        if twin is canonical:
            continue

        # Merge private if canonical is missing it or has placeholder
        canonical_private = getattr(canonical, "private", None)
        is_placeholder = computation_request_cls is not None and isinstance(
            canonical_private, computation_request_cls
        )

        if (canonical_private is None or is_placeholder) and twin.private is not None:
            object.__setattr__(canonical, "private", twin.private)
            if hasattr(twin, "private_stdout"):
                canonical.private_stdout = getattr(twin, "private_stdout", None)
            if hasattr(twin, "private_stderr"):
                canonical.private_stderr = getattr(twin, "private_stderr", None)
            if hasattr(twin, "private_figures"):
                canonical.private_figures = getattr(twin, "private_figures", None)

        # Merge public if canonical is missing it
        if getattr(canonical, "public", None) is None and getattr(twin, "public", None) is not None:
            object.__setattr__(canonical, "public", twin.public)
            if hasattr(twin, "public_stdout"):
                canonical.public_stdout = getattr(twin, "public_stdout", None)
            if hasattr(twin, "public_stderr"):
                canonical.public_stderr = getattr(twin, "public_stderr", None)
            if hasattr(twin, "public_figures"):
                canonical.public_figures = getattr(twin, "public_figures", None)

    # Now make all variables point to the canonical Twin
    updated_vars = []
    for var_name, var_value, namespace in all_twins:
        if var_value is not canonical:
            namespace[var_name] = canonical
            updated_vars.append(var_name)

    if updated_vars:
        print(
            f"🔄 Synced {len(updated_vars)} variable(s) to canonical Twin: {', '.join(updated_vars)}"
        )
        has_public = canonical.public is not None
        has_private = canonical.private is not None
        print(
            f"   Canonical has: public={'✓' if has_public else '✗'}, private={'✓' if has_private else '✗'}"
        )

    # Return the canonical Twin so caller can use it
    return canonical


class CapturedFigure:
    """Wrapper for captured matplotlib figure data that displays nicely in Jupyter."""

    def __init__(self, fig_data: dict):
        """Initialize from figure data dict containing 'figure' and/or 'png_bytes'."""
        self._data = fig_data

    @property
    def figure(self):
        """Get the matplotlib Figure object (may be stale)."""
        return self._data.get("figure")

    @property
    def png_bytes(self) -> bytes:
        """Get the PNG bytes of the figure."""
        return self._data.get("png_bytes", b"")

    def _repr_png_(self):
        """Return PNG bytes for Jupyter display."""
        return self.png_bytes

    def __repr__(self):
        size = len(self.png_bytes) if self.png_bytes else 0
        has_fig = "yes" if self.figure else "no"
        return f"<CapturedFigure png={size} bytes, figure={has_fig}>"


def _safe_preview(val: Any) -> str:
    """
    Short, safe preview for large/complex objects (e.g., AnnData) used in Twin display.
    """
    try:
        try:
            import anndata  # type: ignore

            if isinstance(val, anndata.AnnData):
                return f"AnnData n_obs={val.n_obs}, n_vars={val.n_vars}"
        except Exception:
            pass

        rep = repr(val)
        if len(rep) > 60:
            rep = rep[:57] + "..."
        return rep
    except Exception as exc:
        return f"<{type(val).__name__} (preview failed: {exc})>"


@dataclass
class Twin(LiveMixin, RemoteData):
    """
    A dual-value object holding private (real) and public (mock/synthetic) data.

    Extends RemoteData with privacy-preserving dual values and optional live sync.

    Perfect for collaborative data science:
    - Share mock data freely for development
    - Request real data access with explicit approval
    - Privacy-preserving by default
    - Optional live sync for real-time updates

    Examples:
        >>> data = Twin(private="real.csv", public="mock.csv")
        >>> bv.remote_vars["dataset"] = data  # Only public shared!

        >>> # Remote side
        >>> twin = bv.peer("alice").remote_vars["dataset"]
        >>> result = analyze(twin)  # Uses public side
        >>> result.request_private()  # Request real data

        >>> # Enable live sync
        >>> data.enable_live(mutable=False, interval=2.0)
    """

    # Twin-specific IDs (in addition to base id)
    twin_id: str = field(default_factory=lambda: uuid4().hex)
    private_id: str = field(default_factory=lambda: uuid4().hex)
    public_id: str = field(default_factory=lambda: uuid4().hex)

    # Dual values
    private: Optional[Any] = None  # Real data (only if you own it)
    public: Optional[Any] = None  # Mock/synthetic data (always available)

    # Live sync metadata (survives serialization for display)
    live_enabled: bool = False
    live_interval: float = 2.0

    # Captured outputs from computation execution (must be dataclass fields for pyfory serialization)
    private_stdout: Optional[str] = None
    private_stderr: Optional[str] = None
    private_figures: Optional[list] = None  # List of dicts with _beaver_figure keys
    public_stdout: Optional[str] = None
    public_stderr: Optional[str] = None
    public_figures: Optional[list] = None  # List of dicts with _beaver_figure keys

    # Optional dataset linkage (for auto-loading Twins from published datasets)
    syft_url: Optional[str] = None
    dataset_asset: Optional[str] = None

    def __post_init__(self):
        """Initialize Twin and validate."""
        # Initialize LiveMixin
        super().__post_init__()

        # Set base id to twin_id for consistency
        if not hasattr(self, "id") or self.id == uuid4().hex:
            object.__setattr__(self, "id", self.twin_id)

        # Use object.__getattribute__ to avoid triggering our custom __getattribute__
        raw_private = object.__getattribute__(self, "private")
        raw_public = object.__getattribute__(self, "public")

        # Validate at least one side
        if raw_private is None and raw_public is None:
            raise ValueError("Twin must have at least one side (private or public)")

        # Set var_type from data
        if self.var_type == "unknown":
            if raw_private is not None:
                object.__setattr__(self, "var_type", f"Twin[{type(raw_private).__name__}]")
            elif raw_public is not None:
                object.__setattr__(self, "var_type", f"Twin[{type(raw_public).__name__}]")

        # Internal state (not serialized)
        self._value_accessed = False
        self._last_value_print = 0.0
        self._loading = False  # Prevents recursion during TrustedLoader resolution

        # Live sync subscribers (users who are watching this Twin)
        self._live_subscribers: list[str] = []  # List of usernames
        self._live_context = None  # BeaverContext for sending updates

        # Source tracking for subscribers (where to reload from)
        self._source_path: Optional[str] = None  # Path to reload published data from

        # Register/merge in global registry
        key = (self.twin_id, self.owner)
        existing = _TWIN_REGISTRY.get(key)
        if existing and existing is not self:
            try:
                from .computation import (
                    ComputationRequest as computation_request_cls,  # type: ignore  # noqa: N813
                )
            except Exception:
                computation_request_cls = None  # type: ignore

            # Use object.__getattribute__ to bypass auto-load hook during merge
            existing_pub = object.__getattribute__(existing, "public")
            self_pub = object.__getattribute__(self, "public")

            # Helper to check if value is a stale TrustedLoader (file doesn't exist)
            def _is_stale_trusted_loader(val):
                if isinstance(val, dict) and val.get("_trusted_loader") is True:
                    from pathlib import Path

                    data_path = val.get("path")
                    if data_path and not Path(data_path).exists():
                        return True
                return False

            # Don't merge stale TrustedLoader dicts - prefer fresh data
            if _is_stale_trusted_loader(existing_pub) and self_pub is not None:
                object.__setattr__(existing, "public", self_pub)
            # Merge public if existing lacks it
            elif existing_pub is None and self_pub is not None:
                object.__setattr__(existing, "public", self_pub)
                # Only overwrite captured outputs if new value is not None
                # (preserve existing values during _prepare_for_sending merge)
                for attr in ("public_stdout", "public_stderr", "public_figures"):
                    new_val = getattr(self, attr, None)
                    if new_val is not None:
                        setattr(existing, attr, new_val)

            # Merge private if existing lacks it or only has placeholder request
            existing_priv = object.__getattribute__(existing, "private")
            is_placeholder = computation_request_cls is not None and isinstance(
                existing_priv, computation_request_cls
            )
            is_stale_priv = _is_stale_trusted_loader(existing_priv)

            # Check for idempotent reload (same result type loaded twice)
            self_priv = object.__getattribute__(self, "private")
            is_idempotent = False
            if (
                existing_priv is not None
                and self_priv is not None
                and (
                    (
                        isinstance(existing_priv, CapturedFigure)
                        and isinstance(self_priv, CapturedFigure)
                    )
                    or (type(existing_priv).__name__ == type(self_priv).__name__)
                )
            ):
                is_idempotent = True

            if (
                existing_priv is None or is_placeholder or is_idempotent or is_stale_priv
            ) and self_priv is not None:
                object.__setattr__(existing, "private", self_priv)
                # Only overwrite captured outputs if new value is not None
                # (preserve existing values during _prepare_for_sending merge)
                for attr in ("private_stdout", "private_stderr", "private_figures"):
                    new_val = getattr(self, attr, None)
                    if new_val is not None:
                        setattr(existing, attr, new_val)

            # Keep registry pointing at the original instance
            _TWIN_REGISTRY[key] = existing
        else:
            _TWIN_REGISTRY[key] = self

    def __getattribute__(self, name: str):
        """Intercept access to public/private to handle special cases.

        SECURITY: This hook does NOT auto-execute TrustedLoader code.
        Users must explicitly call .load() to execute loader code.
        This prevents unexpected code execution from simple attribute access.
        """
        # Get the raw value first (avoid infinite recursion)
        value = object.__getattribute__(self, name)

        # Only intercept public and private attributes
        if name in ("public", "private"):
            # Skip if we're in a loading operation (prevents recursion)
            try:
                if object.__getattribute__(self, "_loading"):
                    return value
            except AttributeError:
                pass

            # Handle inline parquet blobs (e.g., decrypted trusted loader payloads)
            # This is safe - just deserializing data bytes, no code execution
            inline = Twin._load_inline_parquet(value)
            if inline is not None:
                object.__setattr__(self, name, inline)
                return inline

            # SECURITY: TrustedLoader dicts are returned as-is
            # Users must explicitly call .load() to execute loader code
            # This prevents unexpected code execution from attribute access

        return value

    def __getstate__(self):
        """Control what gets serialized - exclude non-serializable attributes."""
        state = self.__dict__.copy()
        # Remove non-serializable attributes (internal state)
        non_serializable = [
            "_value_accessed",
            "_last_value_print",  # internal display state
            "_loading",  # TrustedLoader recursion guard
            "_live_subscribers",
            "_live_context",  # runtime state
            "_source_path",
            "_watcher",
            "_live_enabled",
            "_live_mutable",  # live sync state
            "_public_raw",  # raw AnnData reference
        ]
        for attr in non_serializable:
            state.pop(attr, None)

        # Handle figures - convert CapturedFigure objects to serializable dicts
        for fig_attr in ("public_figures", "private_figures"):
            if fig_attr in state:
                figs = state[fig_attr]
                if figs and isinstance(figs, list):
                    converted = []
                    for fig in figs:
                        if isinstance(fig, dict):
                            converted.append(fig)
                        elif isinstance(fig, CapturedFigure) or hasattr(fig, "png_bytes"):
                            converted.append({"_beaver_figure": True, "png_bytes": fig.png_bytes})
                    state[fig_attr] = converted if converted else None

        return state

    def __setstate__(self, state):
        """Restore from serialized state."""
        self.__dict__.update(state)
        # Re-initialize runtime attributes
        self._value_accessed = False
        self._last_value_print = 0.0
        self._loading = False
        self._live_subscribers = []
        self._live_context = None
        self._source_path = None

    @classmethod
    def public_only(
        cls,
        public: Any,
        owner: Optional[str] = None,
        live_enabled: bool = False,
        live_interval: float = 2.0,
        **kwargs,
    ):
        """
        Create a Twin with only public data (no private).

        Used when receiving a Twin from remote.
        """
        from .remote_data import get_current_owner

        return cls(
            private=None,
            public=public,
            owner=owner if owner is not None else get_current_owner(),
            live_enabled=live_enabled,
            live_interval=live_interval,
            **kwargs,
        )

    # ===================================================================
    # RemoteData abstract method implementations
    # ===================================================================

    def has_data(self) -> bool:
        """Check if data is available locally."""
        return self.private is not None or self.public is not None

    def get_value(self) -> Any:
        """Get the preferred value (same as .value property)."""
        return self.value

    # ===================================================================
    # Twin-specific properties
    # ===================================================================

    @property
    def has_private(self) -> bool:
        """Check if private side is available."""
        return self._get_raw_private() is not None

    @property
    def has_public(self) -> bool:
        """Check if public side is available."""
        return self._get_raw_public() is not None

    @property
    def private_value(self):
        """Get private value (raises if not available)."""
        if self.private is None:
            raise ValueError(
                "Private side not available. Call .request_private() to request access from owner."
            )
        return self.private

    @property
    def public_raw(self):
        """Access the underlying public object without any wrappers."""
        return self.public

    @property
    def private_raw(self):
        """Access the underlying private object without any wrappers."""
        return self.private

    @property
    def public_value(self):
        """Get public value (raises if not available)."""
        if self.public is None:
            raise ValueError("Public side not available")
        return self.public

    @property
    def public_safe(self):
        """Get public value with sparse shapes ensured (for repr stability)."""
        try:
            from .remote_vars import _ensure_sparse_shapes

            return _ensure_sparse_shapes(self.public)
        except Exception:
            return self.public

    def can_send(self, **kwargs):
        """
        Check if this Twin can be serialized/deserialized for sending.

        Returns:
            (ok, message)
        """
        from .runtime import can_send as _can_send

        return _can_send(self, **kwargs)

    def show_figures(self, which: str = "public"):
        """
        Display captured matplotlib figures in Jupyter.

        Args:
            which: "public", "private", or "both"
        """
        try:
            from IPython.display import Image, display
        except ImportError:
            print("⚠️  Requires IPython for figure display")
            return

        def display_fig_list(figs, label):
            if not figs:
                print(f"📊 No {label} figures captured")
                return
            print(f"📊 {label.capitalize()} figures ({len(figs)}):")
            for i, fig_data in enumerate(figs):
                if isinstance(fig_data, CapturedFigure):
                    # Display CapturedFigure directly (has _repr_png_)
                    display(fig_data)
                elif isinstance(fig_data, dict) and "png_bytes" in fig_data:
                    # Display from PNG bytes dict
                    display(Image(data=fig_data["png_bytes"]))
                elif hasattr(fig_data, "savefig"):
                    # It's a matplotlib figure object
                    display(fig_data)
                else:
                    print(f"  Figure {i}: {type(fig_data)}")

        if which in ("public", "both"):
            figs = getattr(self, "public_figures", None) or []
            if figs or which == "public":
                display_fig_list(figs, "public")

        if which in ("private", "both"):
            figs = getattr(self, "private_figures", None) or []
            if figs or which == "private":
                display_fig_list(figs, "private")

    def show_output(self, which: str = "public"):
        """
        Display captured stdout/stderr.

        Args:
            which: "public", "private", or "both"
        """
        if which in ("public", "both"):
            stdout = getattr(self, "public_stdout", None)
            stderr = getattr(self, "public_stderr", None)
            if stdout or stderr:
                print("📤 Public output:")
                if stdout:
                    print(f"  stdout: {stdout}")
                if stderr:
                    print(f"  \033[31mstderr: {stderr}\033[0m")
            elif which == "public":
                print("📤 No public output captured")

        if which in ("private", "both"):
            stdout = getattr(self, "private_stdout", None)
            stderr = getattr(self, "private_stderr", None)
            if stdout or stderr:
                print("📤 Private output:")
                if stdout:
                    print(f"  stdout: {stdout}")
                if stderr:
                    print(f"  \033[31mstderr: {stderr}\033[0m")
            elif which == "private":
                print("📤 No private output captured")

    def _is_trusted_loader(self, val: Any) -> bool:
        """Check if a value is an unresolved TrustedLoader dict."""
        return isinstance(val, dict) and val.get("_trusted_loader") is True

    @staticmethod
    def _is_inline_parquet_bytes(val: Any) -> bool:
        """Detect inline parquet payloads (bytes starting with magic PAR1)."""
        return isinstance(val, (bytes, bytearray)) and val[:4] == b"PAR1"

    @staticmethod
    def _load_inline_parquet(val: Any):
        """Attempt to deserialize inline parquet bytes to a pandas object."""
        if not Twin._is_inline_parquet_bytes(val):
            return None
        try:
            import io

            import pandas as pd  # type: ignore

            return pd.read_parquet(io.BytesIO(val), engine="pyarrow")
        except Exception as exc:  # pragma: no cover - runtime env
            print(f"⚠️  Failed to deserialize inline parquet payload: {exc}")
            return None

    def _get_raw_public(self):
        """Get raw public value without triggering __getattribute__ interception."""
        return object.__getattribute__(self, "public")

    def _get_raw_private(self):
        """Get raw private value without triggering __getattribute__ interception."""
        return object.__getattribute__(self, "private")

    # Built-in lib-support deserializer function names (auto-accept these)
    _BUILTIN_DESERIALIZERS = frozenset(
        [
            "pandas_deserialize_file",
            "numpy_deserialize_file",
            "anndata_deserialize_file",
            "pil_deserialize_file",
            "safetensors_deserialize_file",
            "torch_deserialize_file",
        ]
    )

    def _is_builtin_deserializer(self, src: str) -> bool:
        """Check if the deserializer source is a known built-in lib-support function."""
        if not src:
            return False
        first_line = src.strip().split("\n")[0]
        return any(f"def {name}(" in first_line for name in self._BUILTIN_DESERIALIZERS)

    def _resolve_trusted_loader(
        self, loader_dict: dict, auto_accept: bool = False, trust_loader: bool | None = None
    ) -> Any:
        """
        Resolve a TrustedLoader dict by delegating to runtime sandboxed resolver.
        """
        from .runtime import _resolve_trusted_loader

        backend = getattr(self, "_backend", None)
        envelope_path = getattr(self, "_source_path", None)
        return _resolve_trusted_loader(
            loader_dict,
            auto_accept=auto_accept,
            backend=backend,
            trust_loader=trust_loader,
            envelope_path=envelope_path,
        )

    def load(
        self, which: str = "auto", auto_accept: bool = False, trust_loader: bool | None = None
    ) -> Any:
        """
        Load data from TrustedLoader descriptors.

        If public/private contain TrustedLoader dicts, this method resolves them
        by executing the deserializer and replaces the dict with actual data.

        Args:
            which: "auto" (prefer private), "public", "private", or "both"
            auto_accept: If True, skip approval prompts
            trust_loader: If True, run loader in trusted mode (full builtins).
                         If False, use RestrictedPython only (may fail).
                         If None (default), try RestrictedPython first and prompt on failure.

        Returns:
            The loaded value (prefers private if both loaded)

        Example:
            >>> dataset = bv.datasets["owner"]["name"]
            >>> adata = dataset.sc_rnaseq.load()  # prompts, loads, returns AnnData
            >>> # Or explicitly trust for automation:
            >>> adata = dataset.sc_rnaseq.load(trust_loader=True)
        """
        loaded_private = False
        loaded_public = False

        # Use raw accessors to avoid triggering __getattribute__ interception
        raw_private = self._get_raw_private()
        raw_public = self._get_raw_public()

        # Determine what to load
        load_private = which in ("auto", "private", "both") and self._is_trusted_loader(raw_private)
        load_public = which in ("auto", "public", "both") and self._is_trusted_loader(raw_public)

        # If auto and private is a loader, load it
        if load_private:
            print(f"🔒 Loading PRIVATE data for Twin '{self.name or self.twin_id[:8]}'")
            resolved = self._resolve_trusted_loader(
                raw_private, auto_accept=auto_accept, trust_loader=trust_loader
            )
            if resolved is not None:
                object.__setattr__(self, "private", resolved)
                loaded_private = True

        # If auto and public is a loader (and we didn't load private, or explicitly requested)
        if load_public and (not loaded_private or which in ("public", "both")):
            print(f"🌍 Loading PUBLIC data for Twin '{self.name or self.twin_id[:8]}'")
            resolved = self._resolve_trusted_loader(
                raw_public, auto_accept=auto_accept, trust_loader=trust_loader
            )
            if resolved is not None:
                object.__setattr__(self, "public", resolved)
                loaded_public = True

        # Return the preferred value (or None if user cancelled)
        raw_private = self._get_raw_private()  # Re-fetch after potential update
        raw_public = self._get_raw_public()

        # Inline parquet fallback: if we received raw bytes, attempt to decode
        if Twin._is_inline_parquet_bytes(raw_private):
            maybe = Twin._load_inline_parquet(raw_private)
            if maybe is not None:
                object.__setattr__(self, "private", maybe)
                raw_private = maybe
        if Twin._is_inline_parquet_bytes(raw_public):
            maybe = Twin._load_inline_parquet(raw_public)
            if maybe is not None:
                object.__setattr__(self, "public", maybe)
                raw_public = maybe

        if loaded_private or (raw_private is not None and not self._is_trusted_loader(raw_private)):
            return raw_private
        elif loaded_public or (raw_public is not None and not self._is_trusted_loader(raw_public)):
            return raw_public
        else:
            # User cancelled or nothing to load - return None quietly
            return None

    @property
    def value(self):
        """
        Get the preferred value.

        Prefers private if available (local), else public.
        If the value is an unresolved TrustedLoader, prompts to load it.
        """
        import time

        # Initialize if not present (deserialized objects)
        if not hasattr(self, "_value_accessed"):
            self._value_accessed = False
            self._last_value_print = 0.0

        # Use raw accessors to avoid triggering __getattribute__ interception
        raw_private = self._get_raw_private()
        raw_public = self._get_raw_public()

        # Check if we need to resolve TrustedLoader
        preferred = raw_private if raw_private is not None else raw_public
        if self._is_trusted_loader(preferred):
            print(f"\n⚠️  Twin '{self.name or self.twin_id[:8]}' has unloaded data.")
            print("   Call .load() to load it, or approve now:")
            return self.load()

        # Debounced printing: once per evaluation (1 second debounce)
        current_time = time.time()
        if current_time - self._last_value_print > 1.0:
            if raw_private is not None:
                print(f"🔒 Using PRIVATE data from Twin '{self.name or self.twin_id[:8]}...'")
            elif raw_public is not None:
                print(f"🌍 Using PUBLIC data from Twin '{self.name or self.twin_id[:8]}...'")
            self._last_value_print = current_time

        if raw_private is not None:
            return raw_private
        elif raw_public is not None:
            return raw_public
        else:
            raise ValueError("No data available")

    def merge_result(self, result_twin: Twin):
        """
        Merge a received result Twin's properties into this Twin.

        Used when manually loading computation results from inbox.
        The result_twin typically has public=None and private=actual_result.
        This merges the private side and any captured outputs without
        creating a nested Twin structure.

        Args:
            result_twin: The received Twin containing the result

        Example:
            >>> # After manually loading from inbox:
            >>> result_env = bv.inbox()['my_result']
            >>> received = result_env.load(inject=False)
            >>> my_result.merge_result(received)  # Merge into existing Twin
        """
        if not isinstance(result_twin, Twin):
            raise TypeError(f"Expected Twin, got {type(result_twin).__name__}")

        # Merge the private value
        if result_twin.has_private:
            object.__setattr__(self, "private", result_twin.private)

        # Copy over captured outputs
        for attr in ("private_stdout", "private_stderr", "private_figures"):
            if hasattr(result_twin, attr):
                val = getattr(result_twin, attr)
                if val is not None:
                    setattr(self, attr, val)

        # Also update public if the result has it (unusual but possible)
        if result_twin.has_public:
            object.__setattr__(self, "public", result_twin.public)
            for attr in ("public_stdout", "public_stderr", "public_figures"):
                if hasattr(result_twin, attr):
                    val = getattr(result_twin, attr)
                    if val is not None:
                        setattr(self, attr, val)

        print(f"✅ Merged result into Twin '{self.name or self.twin_id[:8]}...'")
        if result_twin.has_private:
            print("   .private now contains the approved result")

    def request_private(self, context=None, name=None):
        """
        Request access to the private side from the owner.

        For data Twins: Creates a request for the owner to share private data.
        For result Twins: Sends the computation request to execute on private data.

        Args:
            context: BeaverContext to use for sending (auto-detected if None)
            name: Override variable name (auto-detected from caller if None)
        """
        import inspect

        from .computation import ComputationRequest as computation_request_cls  # noqa: N813
        from .runtime import _get_var_name_from_caller, pack, write_envelope

        # Check if private is a ComputationRequest (result Twin)
        if isinstance(self.private, computation_request_cls):
            # Auto-detect context if not provided
            if context is None:
                frame = inspect.currentframe()
                if frame and frame.f_back:
                    caller_locals = frame.f_back.f_locals
                    caller_globals = frame.f_back.f_globals
                    # Look for BeaverContext in caller's scope
                    for scope in [caller_locals, caller_globals]:
                        if context:
                            break
                        for _var_name, var_obj in scope.items():
                            if (
                                hasattr(var_obj, "remote_vars")
                                and hasattr(var_obj, "user")
                                and hasattr(var_obj, "inbox_path")
                            ):
                                context = var_obj
                                break

            if not context:
                print("⚠️  No BeaverContext found. Pass context= parameter")
                return

            # Determine the name to use
            # Priority: 1. Explicit name param, 2. Auto-detected variable, 3. Twin's name
            result_name = name
            if result_name is None:
                result_name = _get_var_name_from_caller(self, depth=2)
            if result_name is None:
                result_name = self.name

            print(f"📨 Sending computation request to {self.owner}")
            print(f"   Function: {self.private.func.__name__}")
            print(f"   Result: {result_name}")

            # Pack and send the computation request
            default_request_name = f"request_{self.private.func.__name__}"
            if result_name:
                default_request_name += f"_for_{result_name}"

            # Extract input Twin names from computation args
            input_names = []
            for arg in self.private.args:
                if isinstance(arg, dict) and arg.get("_beaver_remote_var"):
                    if arg.get("name"):
                        input_names.append(arg["name"])
                    elif arg.get("id"):
                        input_names.append(arg["id"][:12])

            env = pack(
                self.private,
                sender=context.user,
                name=default_request_name,
                inputs=input_names,
            )

            # Helper to convert envelope to bytes for encrypted writes
            def env_to_bytes(envelope):
                import json

                from .runtime import _envelope_record

                return json.dumps(_envelope_record(envelope), indent=2).encode("utf-8")

            # Determine destination: use session folder if available, otherwise fall back
            backend = getattr(context, "_backend", None)

            # Get session from Twin or from context's active session
            session = getattr(self, "_session", None)
            if session is None:
                session = getattr(context, "_active_session", None)

            if session is None:
                print("❌ No active session found.")
                print("   Computation requests must be sent within a session.")
                print("   💡 Create or join a session first using the BioVault app.")
                return

            # Write to our session folder (peer can read it via sync)
            dest_dir = session.local_folder
            # Use backend's write_envelope for encryption if available
            if backend and backend.uses_crypto:
                dest_path = dest_dir / env.filename()
                backend.storage.write_with_shadow(
                    absolute_path=str(dest_path),
                    data=env_to_bytes(env),
                    recipients=[self.owner],
                    hint="computation-request",
                    overwrite=True,
                )
                path = dest_path
            else:
                path = write_envelope(env, out_dir=dest_dir)

            print(f"✓ Sent to {path}")
            print(f"💡 Result will auto-update when {self.owner} approves")

            # Store comp_id for auto-update tracking
            self._pending_comp_id = self.private.comp_id

            # Register this Twin for auto-updates
            # Use the canonical Twin from registry (might be different from self due to __post_init__ merge)
            from .computation import _register_twin_result

            key = (self.twin_id, self.owner)
            canonical_twin = _TWIN_REGISTRY.get(key, self)
            _register_twin_result(self.private.comp_id, canonical_twin)

        else:
            # Data Twin - request access to private data
            print(f"🔒 Requesting private data access from {self.owner}")
            print(f"   Twin: {self.name or self.twin_id[:8]}")
            print("   💡 (Data access request flow not yet fully implemented)")

    def subscribe_live(self, user: str, context=None):
        """
        Add a user as a live subscriber to this Twin.

        When Live sync is enabled, changes will automatically be sent to subscribers.

        Args:
            user: Username to subscribe
            context: BeaverContext for sending updates (auto-detected if None)
        """
        # Initialize if not present (deserialized objects)
        if not hasattr(self, "_live_subscribers"):
            self._live_subscribers = []
        if not hasattr(self, "_live_context"):
            self._live_context = None

        # Auto-detect context if not provided
        if context is None and self._live_context is None:
            import inspect

            frame = inspect.currentframe()
            while frame and frame.f_back:
                frame = frame.f_back
                for scope in [frame.f_locals, frame.f_globals]:
                    for _var_name, var_obj in scope.items():
                        if (
                            hasattr(var_obj, "remote_vars")
                            and hasattr(var_obj, "user")
                            and hasattr(var_obj, "inbox_path")
                        ):
                            self._live_context = var_obj
                            break
                    if self._live_context:
                        break
                if self._live_context:
                    break

        if context:
            self._live_context = context

        if not self._live_context:
            print("⚠️  Could not find BeaverContext - pass context= parameter")
            return

        if user not in self._live_subscribers:
            self._live_subscribers.append(user)
            print(f"📡 Added '{user}' as live subscriber to Twin '{self.name}'")
            print("💡 Enable live sync to start sending updates: .enable_live()")
        else:
            print(f"⚠️  '{user}' is already subscribed")

    def unsubscribe_live(self, user: str):
        """
        Remove a user from live subscriptions.

        Args:
            user: Username to unsubscribe
        """
        if not hasattr(self, "_live_subscribers"):
            return

        if user in self._live_subscribers:
            self._live_subscribers.remove(user)
            print(f"🔕 Removed '{user}' from live subscribers")
        else:
            print(f"⚠️  '{user}' is not subscribed")

    def watch_live(self, context=None, interval: float = 1.0):
        """
        Watch for live updates from the owner.

        Continuously monitors the inbox for updates to this Twin and
        automatically reloads when changes arrive.

        Args:
            context: BeaverContext to watch (auto-detected if None)
            interval: Check interval in seconds

        Returns:
            Generator that yields the Twin whenever it updates
        """
        import threading
        import time

        # Auto-detect context
        if context is None:
            import inspect

            frame = inspect.currentframe()
            if frame and frame.f_back:
                for scope in [frame.f_back.f_locals, frame.f_back.f_globals]:
                    for _var_name, var_obj in scope.items():
                        if (
                            hasattr(var_obj, "remote_vars")
                            and hasattr(var_obj, "user")
                            and hasattr(var_obj, "inbox_path")
                        ):
                            context = var_obj
                            break
                    if context:
                        break

        if not context:
            print("⚠️  Could not find BeaverContext - pass context= parameter")
            return

        print(f"👁️  Watching for live updates to Twin '{self.name}'...")
        print(f"💡 Updates from '{self.owner}' will auto-reload")

        # Track seen envelope IDs to avoid re-processing
        seen_envelope_ids = set()
        stop_watching = threading.Event()

        try:
            while not stop_watching.is_set():
                # Check inbox for updates
                try:
                    for envelope in context.inbox():
                        # Check if this is an update to our Twin
                        if envelope.name == self.name and envelope.sender == self.owner:
                            # Skip if we've already processed this envelope
                            if envelope.envelope_id in seen_envelope_ids:
                                continue

                            # Mark as seen
                            seen_envelope_ids.add(envelope.envelope_id)

                            # Load the update
                            updated_twin = envelope.load(inject=False)
                            if isinstance(updated_twin, Twin):
                                # Update our Twin's data
                                if updated_twin.has_public:
                                    self.public = updated_twin.public
                                if updated_twin.has_private:
                                    self.private = updated_twin.private

                                # Yield the updated Twin
                                yield self
                except Exception:
                    # Don't crash on inbox errors
                    pass

                # Sleep before next check
                time.sleep(interval)

        except KeyboardInterrupt:
            print(f"\n⚫ Stopped watching Twin '{self.name}'")

    # ===================================================================
    # Delegation methods
    # ===================================================================

    def __bool__(self):
        """Always truthy; prevents bool() from calling __len__ on non-sized values."""
        return True

    def __len__(self):
        """Delegate len() to preferred value."""
        try:
            return len(self.value)
        except (TypeError, ValueError):
            # Fall back to 1 so truthiness checks don't explode on figure/None
            return 1

    def __getitem__(self, key):
        """Delegate indexing to preferred value."""
        return self.value[key]

    # ===================================================================
    # Display methods (unified with RemoteData)
    # ===================================================================

    def __str__(self) -> str:
        """String representation showing both sides with visual state indicators."""
        # Use raw accessors to check for TrustedLoader without triggering resolution
        raw_private = self._get_raw_private()
        raw_public = self._get_raw_public()

        # Check if this is a dataset asset Twin with TrustedLoader
        has_loader = self._is_trusted_loader(raw_private) or self._is_trusted_loader(raw_public)
        if has_loader:
            return self._str_trusted_loader(raw_private, raw_public)

        lines = []
        twin_name = self.name or f"Twin#{self.twin_id[:8]}"

        # Determine state and color scheme
        # Private only (red) = real sensitive data
        # Public only (green) = safe mock data
        # Both (yellow/gold) = be careful which you use
        # Neither/pending (purple) = waiting for data

        if self.has_private and self.has_public:
            # Both sides - yellow/gold warning
            header = f"⚠️  Twin: {twin_name} \033[33m(REAL + MOCK DATA)\033[0m"
            warning = "  ⚠️  \033[33mBe careful: This Twin contains both real and mock data\033[0m"
            lines.append(header)
            lines.append(warning)
        elif self.has_private:
            # Private only - red warning
            header = f"🔒 Twin: {twin_name} \033[31m(REAL DATA - SENSITIVE)\033[0m"
            lines.append(header)
        elif self.has_public:
            # Public only - green safe
            header = f"🌍 Twin: {twin_name} \033[32m(MOCK DATA - SAFE)\033[0m"
            lines.append(header)
        else:
            # Neither - purple pending
            header = f"⏳ Twin: {twin_name} \033[35m(PENDING)\033[0m"
            lines.append(header)

        # Private side
        if self.has_private:
            private_repr = _safe_preview(self.private)
            lines.append(f"  \033[31m🔒 Private\033[0m    {private_repr}    ← .value uses this")
        else:
            lines.append("  🔒 Private    (not available) 💡 .request_private()")

        # Public side
        if self.has_public:
            public_repr = _safe_preview(self.public)
            active_marker = "    ← .value uses this" if not self.has_private else "    ✓"
            lines.append(f"  \033[32m🌍 Public\033[0m    {public_repr}{active_marker}")
        else:
            lines.append("  🌍 Public    (not set)")

        # Owner (if not local)
        if not self.has_private:
            lines.append(f"  Owner: {self.owner}")

        # Live status (from LiveMixin)
        if hasattr(self, "_display_live_status"):
            lines.extend(self._display_live_status())

        # Captured outputs (from @bv decorated function execution)
        if hasattr(self, "public_stdout") and self.public_stdout:
            lines.append(f"  📤 Captured stdout: {len(self.public_stdout)} chars")
        if hasattr(self, "public_stderr") and self.public_stderr:
            lines.append(f"  ⚠️  Captured stderr: {len(self.public_stderr)} chars")
        if hasattr(self, "public_figures") and self.public_figures:
            lines.append(f"  📊 Captured figures: {len(self.public_figures)}")

        # IDs (compact)
        lines.append(
            f"  IDs: twin={self.twin_id[:8]}... private={self.private_id[:8]}... public={self.public_id[:8]}..."
        )

        # Hint about captured outputs
        if (
            (hasattr(self, "public_stdout") and self.public_stdout)
            or (hasattr(self, "public_stderr") and self.public_stderr)
            or (hasattr(self, "public_figures") and self.public_figures)
        ):
            lines.append("  💡 Access: .public_stdout, .public_stderr, .public_figures")

        return "\n".join(lines)

    def _str_trusted_loader(self, raw_private, raw_public) -> str:
        """Format display for Twin with TrustedLoader data (dataset assets)."""
        # ANSI colors
        cyan = "\033[36m"
        green = "\033[32m"
        dim = "\033[2m"
        bold = "\033[1m"
        reset = "\033[0m"

        twin_name = self.name or f"Twin#{self.twin_id[:8]}"

        # Get loader info (prefer private, fall back to public)
        loader = raw_private if self._is_trusted_loader(raw_private) else raw_public
        asset_name = loader.get("name", "unknown")
        data_path = loader.get("path", "unknown")
        src = loader.get("deserializer_src", "")
        data_type = loader.get("data_type", "unknown")

        # Determine if private/public are loaders
        has_private_loader = self._is_trusted_loader(raw_private)
        has_public_loader = self._is_trusted_loader(raw_public)

        lines = []
        lines.append(f"{bold}Twin{reset} [{green}DATASET ASSET{reset}] {cyan}{twin_name}{reset}")
        lines.append(f"  {bold}Owner:{reset} {self.owner}")
        lines.append(f"  {bold}Asset:{reset} {asset_name}")
        lines.append(f"  {bold}Type:{reset} {data_type}")
        lines.append(f"  {bold}Path:{reset} {dim}{data_path}{reset}")

        # Show which slots have loaders
        if has_private_loader and has_public_loader:
            lines.append(f"  {bold}Data:{reset} 🔒 Private + 🌍 Public (both unloaded)")
        elif has_private_loader:
            lines.append(f"  {bold}Data:{reset} 🔒 Private (unloaded)")
        elif has_public_loader:
            lines.append(f"  {bold}Data:{reset} 🌍 Public (unloaded)")

        lines.append(f"  {bold}ID:{reset} {dim}{self.twin_id[:8]}...{reset}")

        # Show the loader source code (internal deserializer)
        if src:
            lines.append("")
            lines.append(f"{bold}Deserializer:{reset} {dim}(runs automatically){reset}")
            for i, line in enumerate(src.strip().splitlines(), 1):
                lines.append(f"  {dim}{i:2d}{reset} │ {cyan}{line}{reset}")

        lines.append("")
        lines.append(
            f"💡 {cyan}data = asset.load(){reset}  {dim}# returns the loaded object{reset}"
        )

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Use string representation."""
        return self.__str__()

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter."""
        # Use raw accessors to check for TrustedLoader without triggering resolution
        raw_private = self._get_raw_private()
        raw_public = self._get_raw_public()

        # Check if this is a dataset asset Twin with TrustedLoader
        has_loader = self._is_trusted_loader(raw_private) or self._is_trusted_loader(raw_public)
        if has_loader:
            return self._repr_html_trusted_loader(raw_private, raw_public)

        twin_name = self.name or f"Twin#{self.twin_id[:8]}"

        private_html = ""
        if self.has_private:
            private_repr = _safe_preview(self.private)
            private_html = f"<tr><td>🔒 Private</td><td><code>{private_repr}</code></td><td><span style='color: green; font-weight: bold;'>← .value uses this</span></td></tr>"
        else:
            private_html = "<tr><td>🔒 Private</td><td><i>not available</i></td><td><a href='#'>.request_private()</a></td></tr>"

        public_html = ""
        if self.has_public:
            public_repr = _safe_preview(self.public)
            active_marker = (
                "<span style='color: green; font-weight: bold;'>← .value uses this</span>"
                if not self.has_private
                else "✓"
            )
            public_html = f"<tr><td>🌍 Public</td><td><code>{public_repr}</code></td><td>{active_marker}</td></tr>"
        else:
            public_html = "<tr><td>🌍 Public</td><td><i>not set</i></td><td></td></tr>"

        owner_html = ""
        if not self.has_private:
            owner_html = f"<tr><td>Owner</td><td colspan='2'><b>{self.owner}</b></td></tr>"

        # Live status
        live_html = ""
        if hasattr(self, "_live_enabled") and self._live_enabled:
            mode = "mutable" if self._live_mutable else "read-only"
            live_html = f"<tr><td>Live</td><td colspan='2'>🟢 Enabled ({mode}, {self._live_interval}s)</td></tr>"
        else:
            live_html = "<tr><td>Live</td><td colspan='2'>⚫ Disabled</td></tr>"

        # Choose border color based on Twin state
        # Red = real/private data (sensitive)
        # Green = public/mock only (safe)
        # Yellow/Orange = both (be careful)
        # Purple = pending/neither (waiting)
        if self.has_private and self.has_public:
            border_color = "#FF9800"  # Orange (warning - both sides)
        elif self.has_private:
            border_color = "#F44336"  # Red (danger - real data)
        elif self.has_public:
            border_color = "#4CAF50"  # Green (safe - mock data)
        else:
            border_color = "#9C27B0"  # Purple (pending)

        return f"""
        <div style='font-family: monospace; border-left: 3px solid {border_color}; padding-left: 10px;'>
            <b>Twin:</b> {twin_name}<br>
            <table style='margin-top: 10px;'>
                {private_html}
                {public_html}
                {owner_html}
                {live_html}
                <tr><td colspan='3' style='padding-top: 10px; font-size: 10px; color: #666;'>
                    IDs: twin={self.twin_id[:8]}... private={self.private_id[:8]}... public={self.public_id[:8]}...
                </td></tr>
            </table>
        </div>
        """

    def _repr_html_trusted_loader(self, raw_private, raw_public) -> str:
        """HTML representation for Twin with TrustedLoader data (dataset assets)."""
        import html as html_module

        twin_name = self.name or f"Twin#{self.twin_id[:8]}"

        # Get loader info (prefer private, fall back to public)
        loader = raw_private if self._is_trusted_loader(raw_private) else raw_public
        asset_name = loader.get("name", "unknown")
        data_path = loader.get("path", "unknown")
        src = loader.get("deserializer_src", "")
        data_type = loader.get("data_type", "unknown")

        # Determine if private/public are loaders
        has_private_loader = self._is_trusted_loader(raw_private)
        has_public_loader = self._is_trusted_loader(raw_public)

        if has_private_loader and has_public_loader:
            data_status = "🔒 Private + 🌍 Public (both unloaded)"
        elif has_private_loader:
            data_status = "🔒 Private (unloaded)"
        else:
            data_status = "🌍 Public (unloaded)"

        # Format source code with syntax highlighting
        source_html = ""
        if src:
            try:
                from pygments import highlight
                from pygments.formatters import HtmlFormatter
                from pygments.lexers import PythonLexer

                formatter = HtmlFormatter(style="monokai", noclasses=True)
                highlighted = highlight(src, PythonLexer(), formatter)
                source_html = f"""
                <br><br><b>Deserializer:</b> <span style='color: #666; font-size: 11px;'>(runs automatically)</span>
                <div style='border: 1px solid rgba(128, 128, 128, 0.3);
                            border-radius: 4px; overflow: hidden; margin-top: 5px;'>{highlighted}</div>
                """
            except ImportError:
                # Fallback if pygments not available
                escaped_source = html_module.escape(src)
                source_html = f"""
                <br><br><b>Deserializer:</b> <span style='color: #666; font-size: 11px;'>(runs automatically)</span>
                <pre style='background: rgba(128, 128, 128, 0.1);
                            border: 1px solid rgba(128, 128, 128, 0.3);
                            padding: 10px; margin-top: 5px; border-radius: 4px;'>{escaped_source}</pre>
                """

        return f"""
        <div style='font-family: monospace; border-left: 3px solid #4CAF50; padding-left: 10px;'>
            <b>Twin</b> <span style='background: #4CAF50; color: white; padding: 2px 6px;
                                     border-radius: 3px; font-size: 10px;'>DATASET ASSET</span>
            <code>{html_module.escape(twin_name)}</code><br>
            <b>Owner:</b> {html_module.escape(self.owner)}<br>
            <b>Asset:</b> {html_module.escape(asset_name)}<br>
            <b>Type:</b> {html_module.escape(data_type)}<br>
            <b>Path:</b> <code style='font-size: 11px;'>{html_module.escape(data_path)}</code><br>
            <b>Data:</b> {data_status}<br>
            <b>ID:</b> <code>{self.twin_id[:8]}...</code>
            {source_html}
            <br><br><span style='color: #666;'>💡 <code>data = asset.load()</code> <span style='font-size: 11px;'># returns the loaded object</span></span>
        </div>
        """
