from __future__ import annotations

import base64
import collections.abc
import contextlib
import functools
import inspect
import json
import re
import tempfile
import textwrap
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple
from uuid import uuid4

try:
    import pyfory
except ImportError:
    import pickle

    # Minimal stub for environments without pyfory (tests/dev)
    class _StubFory:
        def __init__(self, *args, **kwargs):
            pass

        def register_type(self, *_args, **_kwargs):
            return None

        def dumps(self, obj):
            return pickle.dumps(obj)

        def loads(self, payload):
            return pickle.loads(payload)

    pyfory = types.SimpleNamespace(Fory=_StubFory)

from .envelope import BeaverEnvelope


class TrustedLoader:
    """
    Simple decorator-based trusted loader registry.

    Usage:
        @TrustedLoader.register(MyType)
        def serializer(obj, path): ...

        @TrustedLoader.register(MyType)
        def deserializer(path): ...

    The first decorator sets the serializer, the second sets the deserializer.
    """

    _handlers: Dict[type, Dict[str, Any]] = {}

    @classmethod
    def register(cls, typ: type, *, name: Optional[str] = None):
        def deco(fn):
            entry = cls._handlers.setdefault(
                typ,
                {
                    "name": name or f"{typ.__module__}.{typ.__name__}",
                    "serializer": None,
                    "deserializer": None,
                },
            )
            if entry["serializer"] is None:
                entry["serializer"] = fn
            else:
                entry["deserializer"] = fn
            return fn

        return deco

    @classmethod
    def get(cls, typ: type) -> Optional[Dict[str, Any]]:
        h = cls._handlers.get(typ)
        if h and h.get("serializer") and h.get("deserializer"):
            return h
        return None


def _strip_ansi_codes(text: str) -> str:
    """Remove ANSI color codes from text."""
    # Pattern matches ANSI escape sequences like \033[31m (red) or \033[0m (reset)
    ansi_pattern = re.compile(r"\033\[[0-9;]*m")
    return ansi_pattern.sub("", text)


def _get_var_name_from_caller(obj: Any, depth: int = 2) -> Optional[str]:
    """
    Try to extract the variable name from the caller's frame.

    Args:
        obj: The object being sent
        depth: How many frames to go back (default 2: _get_var_name_from_caller -> send -> caller)

    Returns:
        Variable name if found, else None
    """
    try:
        frame = inspect.currentframe()
        for _ in range(depth):
            if frame is None:
                return None
            frame = frame.f_back

        if frame is None:
            return None

        # Get the calling line of code
        import linecache

        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        line = linecache.getline(filename, lineno).strip()

        # Try to extract variable name from patterns like:
        # bv.send(dataset, ...)
        # result = bv.send(dataset, ...)
        # my_var.request_private()
        # my_var.request_private(context=bv)

        # Match: .send(varname, ...) or .send(varname)
        match = re.search(r"\.send\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)", line)
        if match:
            var_name = match.group(1)
            # Verify this variable exists in the caller's locals and references our object
            if var_name in frame.f_locals and frame.f_locals[var_name] is obj:
                return var_name

        # Match: varname.request_private(...) or varname.method()
        match = re.search(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\.\s*\w+\s*\(", line)
        if match:
            var_name = match.group(1)
            # Verify this variable exists in the caller's locals and references our object
            if var_name in frame.f_locals and frame.f_locals[var_name] is obj:
                return var_name

        # Fallback: check if obj has a __name__ attribute (functions, classes)
        if hasattr(obj, "__name__"):
            return obj.__name__

    except Exception:
        pass

    return None


def _summarize(obj: Any) -> dict:
    # Get the underlying function if it's a partial
    func = obj
    if hasattr(obj, "func"):
        func = obj.func

    # Determine type - show function/class instead of partial if no args bound
    obj_type = type(obj).__name__
    if obj_type == "partial" and hasattr(obj, "func"):
        # Check if any args are actually bound
        has_bound_args = bool(obj.args or obj.keywords)
        if not has_bound_args:
            # No args bound, show as the underlying function type
            obj_type = type(func).__name__

    summary = {
        "type": obj_type,
        "module": getattr(func, "__module__", None) or getattr(obj, "__module__", None),
        "qualname": getattr(func, "__qualname__", None) or getattr(obj, "__qualname__", None),
    }

    # Determine envelope type: code or data
    is_code = inspect.isfunction(func) or inspect.isclass(func) or inspect.ismethod(func)
    summary["envelope_type"] = "code" if is_code else "data"

    if hasattr(obj, "__len__"):
        with contextlib.suppress(Exception):
            summary["len"] = len(obj)  # type: ignore[arg-type]

    # Add data-specific info
    if not is_code:
        with contextlib.suppress(Exception):
            # Capture preview/repr for data envelopes
            preview = repr(obj)
            # Strip ANSI color codes (e.g., from Twin objects)
            preview = _strip_ansi_codes(preview)
            # Limit preview size to 500 chars
            if len(preview) > 500:
                preview = preview[:497] + "..."
            summary["preview"] = preview

            # For numpy arrays
            if hasattr(obj, "shape"):
                summary["shape"] = str(obj.shape)
                summary["dtype"] = str(obj.dtype)
            # For pandas dataframes
            elif hasattr(obj, "columns"):
                summary["columns"] = list(obj.columns)
                summary["shape"] = str(obj.shape)

    # Try to capture source code and signature for code envelopes
    if is_code:
        with contextlib.suppress(Exception):
            # Capture signature for functions
            if inspect.isfunction(func):
                sig = inspect.signature(func)
                summary["signature"] = str(sig)

                # Check for return annotation
                if sig.return_annotation != inspect.Signature.empty:
                    return_type = sig.return_annotation
                    if hasattr(return_type, "__name__"):
                        summary["return_type"] = return_type.__name__
                    else:
                        summary["return_type"] = str(return_type)

            # Capture source code
            source = inspect.getsource(func)
            if source:
                # Strip decorators from source
                lines = source.splitlines()
                cleaned_lines = []
                skip_blank = False

                for line in lines:
                    stripped = line.lstrip()
                    # Skip decorator lines
                    if stripped.startswith("@"):
                        skip_blank = True  # Skip blank line after decorator too
                        continue
                    # Skip one blank line after decorator
                    if skip_blank and not stripped:
                        skip_blank = False
                        continue
                    skip_blank = False
                    cleaned_lines.append(line)

                # Remove leading blank lines
                while cleaned_lines and not cleaned_lines[0].strip():
                    cleaned_lines.pop(0)

                cleaned_source = "\n".join(cleaned_lines)
                summary["source"] = cleaned_source
                summary["source_lines"] = len(cleaned_lines)

    return summary


def _strip_non_serializable_attrs(twin):
    """Remove non-serializable attributes (like matplotlib figures) from a Twin."""
    # These attributes are added by @bv decorator for local display but can't be serialized
    for attr in ("public_figures", "private_figures"):
        if hasattr(twin, attr):
            delattr(twin, attr)
    return twin


def _sanitize_for_serialization(obj: Any) -> Any:
    """Recursively sanitize objects that can't be serialized by pyfory."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool, bytes)):
        return obj

    # Handle matplotlib Figure/Axes
    try:
        import matplotlib.figure
        from matplotlib.axes import Axes

        if isinstance(obj, matplotlib.figure.Figure):
            import io

            buf = io.BytesIO()
            obj.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            buf.seek(0)
            return {"_beaver_figure": True, "png_bytes": buf.getvalue()}
        if isinstance(obj, Axes):
            if obj.figure:
                import io

                buf = io.BytesIO()
                obj.figure.savefig(buf, format="png", bbox_inches="tight", dpi=100)
                buf.seek(0)
                return {"_beaver_figure": True, "png_bytes": buf.getvalue()}
            return {"_beaver_axes": True, "title": str(obj.get_title())}
    except ImportError:
        pass

    # Handle numpy types
    try:
        import numpy as np

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist() if obj.size <= 10000 else {"_numpy_array": True, "shape": obj.shape}
    except ImportError:
        pass

    # Handle pandas
    try:
        import pandas as pd

        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="list")
        if isinstance(obj, pd.Series):
            return obj.to_dict()
    except ImportError:
        pass

    # Handle scipy sparse
    try:
        import scipy.sparse

        if scipy.sparse.issparse(obj):
            return (
                obj.toarray().tolist()
                if obj.nnz < 10000
                else {"_sparse_matrix": True, "shape": obj.shape}
            )
    except ImportError:
        pass

    # Recursively handle collections
    if isinstance(obj, dict):
        return {k: _sanitize_for_serialization(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_serialization(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_sanitize_for_serialization(x) for x in obj)

    # Check for problematic objects
    obj_type = type(obj)
    type_name = f"{obj_type.__module__}.{obj_type.__name__}"
    if "matplotlib" in type_name:
        return {"_matplotlib_object": True, "type": type_name}
    # Check if object lacks __dict__ (can't be serialized properly)
    if not hasattr(obj, "__dict__"):
        # Has __slots__ but might still be problematic
        if hasattr(obj, "__slots__"):
            slots = getattr(obj_type, "__slots__", ())
            if not slots or (isinstance(slots, (list, tuple)) and len(slots) == 0):
                # Empty slots - object has no data
                return {"_unserializable": True, "type": type_name}
        else:
            return {"_unserializable": True, "type": type_name}
    if any(mod in type_name for mod in ("scanpy", "anndata", "sklearn")):
        return {"_complex_object": True, "type": type_name}
    return obj


def _prepare_for_sending(
    obj: Any,
    *,
    artifact_dir: Optional[Path] = None,
    name_hint: Optional[str] = None,
    preserve_private: bool = False,
) -> Any:
    """Prepare object for sending by handling trusted loaders and optionally stripping private data.

    Args:
        obj: Object to prepare
        artifact_dir: Directory for trusted loader artifacts
        name_hint: Name hint for artifacts
        preserve_private: If True, keep private data in Twins (for approved results)
    """
    from .twin import Twin

    # Trusted loader conversion (non-Twin)
    tl = TrustedLoader.get(type(obj))
    if tl:
        target_dir = Path(artifact_dir) if artifact_dir else Path(tempfile.gettempdir())
        target_dir.mkdir(parents=True, exist_ok=True)
        base = name_hint or tl["name"].split(".")[-1] or "artifact"
        path = target_dir / f"{base}.bin"
        tl["serializer"](obj, path)
        import inspect

        src_lines = inspect.getsource(tl["deserializer"]).splitlines()
        src_clean = "\n".join(
            line for line in src_lines if not line.lstrip().startswith("@TrustedLoader")
        )
        src_clean = textwrap.dedent(src_clean)
        return {
            "_trusted_loader": True,
            "name": tl["name"],
            "path": str(path),
            "deserializer_src": src_clean,
        }

    # Handle Twin objects
    if isinstance(obj, Twin):
        public_obj = obj.public
        private_obj = obj.private if preserve_private else None

        # Apply trusted loader to public side if available
        tl_pub = TrustedLoader.get(type(public_obj)) if public_obj is not None else None
        if tl_pub:
            target_dir = Path(artifact_dir) if artifact_dir else Path(tempfile.gettempdir())
            target_dir.mkdir(parents=True, exist_ok=True)
            base = name_hint or getattr(public_obj, "name", None) or "artifact"
            path = target_dir / f"{base}_public.bin"
            tl_pub["serializer"](public_obj, path)
            import inspect

            src_lines = inspect.getsource(tl_pub["deserializer"]).splitlines()
            src_clean = "\n".join(
                line for line in src_lines if not line.lstrip().startswith("@TrustedLoader")
            )
            src_clean = textwrap.dedent(src_clean)
            public_obj = {
                "_trusted_loader": True,
                "name": tl_pub["name"],
                "path": str(path),
                "deserializer_src": src_clean,
            }

        # Apply trusted loader to private side if preserving and available
        if preserve_private and private_obj is not None:
            tl_priv = TrustedLoader.get(type(private_obj))
            if tl_priv:
                target_dir = Path(artifact_dir) if artifact_dir else Path(tempfile.gettempdir())
                target_dir.mkdir(parents=True, exist_ok=True)
                base = name_hint or getattr(private_obj, "name", None) or "artifact"
                path = target_dir / f"{base}_private.bin"
                tl_priv["serializer"](private_obj, path)
                import inspect

                src_lines = inspect.getsource(tl_priv["deserializer"]).splitlines()
                src_clean = "\n".join(
                    line for line in src_lines if not line.lstrip().startswith("@TrustedLoader")
                )
                src_clean = textwrap.dedent(src_clean)
                private_obj = {
                    "_trusted_loader": True,
                    "name": tl_priv["name"],
                    "path": str(path),
                    "deserializer_src": src_clean,
                }

        # Sanitize values before creating Twin to avoid serialization issues
        # Skip if already a trusted loader dict
        if public_obj is not None and not (
            isinstance(public_obj, dict) and public_obj.get("_trusted_loader")
        ):
            public_obj = _sanitize_for_serialization(public_obj)
        if private_obj is not None and not (
            isinstance(private_obj, dict) and private_obj.get("_trusted_loader")
        ):
            private_obj = _sanitize_for_serialization(private_obj)

        # Create new Twin (with or without private)
        if preserve_private and private_obj is not None:
            new_twin = Twin(
                public=public_obj,
                private=private_obj,
                owner=obj.owner,
                twin_id=obj.twin_id,
                private_id=obj.private_id,
                public_id=obj.public_id,
                name=obj.name,
                live_enabled=obj.live_enabled,
                live_interval=obj.live_interval,
            )
        else:
            new_twin = Twin.public_only(
                public=public_obj,
                owner=obj.owner,
                twin_id=obj.twin_id,
                private_id=obj.private_id,
                public_id=obj.public_id,
                name=obj.name,
                live_enabled=obj.live_enabled,
                live_interval=obj.live_interval,
            )

        # Copy serializable captured outputs
        for attr in ("public_stdout", "public_stderr", "private_stdout", "private_stderr"):
            if hasattr(obj, attr):
                setattr(new_twin, attr, getattr(obj, attr))

        # Convert captured figures to serializable format (dicts with PNG bytes)
        def convert_figures_for_sending(figs):
            """Convert CapturedFigure objects to serializable dicts."""
            if not figs:
                return None
            result = []
            for fig_item in figs:
                if hasattr(fig_item, "png_bytes"):
                    # CapturedFigure - extract PNG bytes
                    result.append({"_beaver_figure": True, "png_bytes": fig_item.png_bytes})
                elif isinstance(fig_item, dict) and fig_item.get("_beaver_figure"):
                    # Already a dict, keep as-is
                    result.append(fig_item)
            return result if result else None

        if hasattr(obj, "public_figures") and obj.public_figures:
            new_twin.public_figures = convert_figures_for_sending(obj.public_figures)
        if hasattr(obj, "private_figures") and obj.private_figures:
            new_twin.private_figures = convert_figures_for_sending(obj.private_figures)

        return new_twin

    # Handle collections containing Twins
    if isinstance(obj, dict):
        return {
            k: _prepare_for_sending(
                v, artifact_dir=artifact_dir, name_hint=k, preserve_private=preserve_private
            )
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        result = [
            _prepare_for_sending(item, artifact_dir=artifact_dir, preserve_private=preserve_private)
            for item in obj
        ]
        return type(obj)(result)

    # Return as-is for other types
    return obj


def pack(
    obj: Any,
    *,
    envelope_id: Optional[str] = None,
    sender: str = "unknown",
    name: Optional[str] = None,
    inputs: Optional[Iterable[str]] = None,
    outputs: Optional[Iterable[str]] = None,
    requirements: Optional[Iterable[str]] = None,
    reply_to: Optional[str] = None,
    strict: bool = False,
    policy=None,
    artifact_dir: Optional[Path | str] = None,
    preserve_private: bool = False,
) -> BeaverEnvelope:
    """Serialize an object into a BeaverEnvelope (Python-native).

    Args:
        preserve_private: If True, include private data in Twins (for approved results)
    """
    # Prepare for sending (optionally strip private data)
    obj_to_send = _prepare_for_sending(
        obj, artifact_dir=artifact_dir, name_hint=name, preserve_private=preserve_private
    )

    fory = pyfory.Fory(xlang=False, ref=True, strict=strict, policy=policy)

    # Register Twin type so it can be deserialized
    from .twin import Twin

    fory.register_type(Twin)

    payload = fory.dumps(obj_to_send)
    manifest = _summarize(obj)
    manifest["size_bytes"] = len(payload)
    manifest["language"] = "python"

    env_id = envelope_id or uuid4().hex

    return BeaverEnvelope(
        envelope_id=env_id,
        sender=sender,
        name=name,
        inputs=list(inputs or []),
        outputs=list(outputs or []),
        requirements=list(requirements or []),
        manifest=manifest,
        payload=payload,
        reply_to=reply_to,
    )


def _envelope_record(envelope: BeaverEnvelope) -> dict:
    return {
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


def write_envelope(envelope: BeaverEnvelope, out_dir: Path | str = ".") -> Path:
    """Persist a .beaver file (JSON manifest + base64 payload)."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    path = out_path / envelope.filename()
    path.write_text(json.dumps(_envelope_record(envelope), indent=2))
    return path


def read_envelope(path: Path | str) -> BeaverEnvelope:
    """Load a BeaverEnvelope from disk."""
    record = json.loads(Path(path).read_text())
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


def unpack(
    envelope: BeaverEnvelope,
    *,
    strict: bool = False,
    policy=None,
    auto_accept: bool = False,
) -> Any:
    """Deserialize the payload in a BeaverEnvelope.

    Args:
        envelope: The envelope to unpack
        strict: Strict deserialization mode
        policy: Deserialization policy
        auto_accept: If True, automatically accept trusted loaders without prompting
    """
    _install_builtin_aliases()
    fory = pyfory.Fory(xlang=False, ref=True, strict=strict, policy=policy)

    # Register Twin type so it can be deserialized
    from .twin import Twin

    fory.register_type(Twin)

    obj = fory.loads(envelope.payload)
    obj = _resolve_trusted_loader(obj, auto_accept=auto_accept)
    return obj


def _resolve_trusted_loader(obj: Any, *, auto_accept: bool = False) -> Any:
    """
    Resolve trusted-loader descriptors, prompting before execution.

    Also unwraps _beaver_figure dicts back to CapturedFigure for nice display.

    Args:
        obj: The object to resolve
        auto_accept: If True, automatically accept trusted loaders without prompting
    """
    try:
        from .twin import CapturedFigure, Twin  # local import to avoid cycles

        twin_cls = Twin
        captured_figure_cls = CapturedFigure
    except Exception:
        twin_cls = None  # type: ignore
        captured_figure_cls = None  # type: ignore

    if twin_cls is not None and isinstance(obj, twin_cls):
        obj.public = _resolve_trusted_loader(obj.public, auto_accept=auto_accept)
        obj.private = _resolve_trusted_loader(obj.private, auto_accept=auto_accept)
        # Also resolve captured figure lists (they contain _beaver_figure dicts)
        if hasattr(obj, "public_figures") and obj.public_figures:
            obj.public_figures = _resolve_trusted_loader(
                obj.public_figures, auto_accept=auto_accept
            )
        if hasattr(obj, "private_figures") and obj.private_figures:
            obj.private_figures = _resolve_trusted_loader(
                obj.private_figures, auto_accept=auto_accept
            )
        return obj

    # Unwrap _beaver_figure dicts back to CapturedFigure for nice Jupyter display
    if isinstance(obj, dict) and obj.get("_beaver_figure") and captured_figure_cls is not None:
        return captured_figure_cls(obj)

    if isinstance(obj, dict) and obj.get("_trusted_loader"):
        name = obj.get("name")
        data_path = obj.get("path")
        src = obj.get("deserializer_src", "")
        preview = "\n".join(src.strip().splitlines()[:5])
        if not auto_accept:
            resp = (
                input(
                    f"Execute trusted loader '{name}'? [y/N]:\n{preview}\nSource: {data_path}\nProceed? "
                )
                .strip()
                .lower()
            )
            if resp not in ("y", "yes"):
                raise RuntimeError("Loader not approved")
        # Build scope with common imports that deserializers might need
        scope: Dict[str, Any] = {"TrustedLoader": TrustedLoader}
        # Try to import anndata (common for AnnData loaders)
        try:
            import anndata as ad

            scope["ad"] = ad
            scope["anndata"] = ad
        except ImportError:
            pass
        # Try to import pandas (common for DataFrame loaders)
        try:
            import pandas as pd

            scope["pd"] = pd
            scope["pandas"] = pd
        except ImportError:
            pass
        # Try to import numpy (common dependency)
        try:
            import numpy as np

            scope["np"] = np
            scope["numpy"] = np
        except ImportError:
            pass
        exec(src, scope, scope)
        # Exclude TrustedLoader class and imported modules - we want the actual deserializer function
        excluded = {
            TrustedLoader,
            scope.get("ad"),
            scope.get("pd"),
            scope.get("np"),
            scope.get("anndata"),
            scope.get("pandas"),
            scope.get("numpy"),
        }
        deser_fn = next((v for v in scope.values() if callable(v) and v not in excluded), None)
        if deser_fn is None:
            raise RuntimeError("No deserializer found in loader source")
        return deser_fn(data_path)

    if isinstance(obj, list):
        return [_resolve_trusted_loader(x, auto_accept=auto_accept) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_resolve_trusted_loader(x, auto_accept=auto_accept) for x in obj)
    if isinstance(obj, dict):
        return {k: _resolve_trusted_loader(v, auto_accept=auto_accept) for k, v in obj.items()}
    return obj


def can_send(obj: Any, *, strict: bool = False, policy=None) -> tuple[bool, str]:
    """
    Dry-run serialization/deserialization to test sendability.

    Uses auto_accept=True since this is testing your own data - no need to
    prompt for trusted loader approval on a dry-run test.
    """
    try:
        env = pack(obj, strict=strict, policy=policy)
        _ = unpack(env, strict=strict, policy=policy, auto_accept=True)
        return True, f"ok: {type(obj).__name__} serializes"
    except Exception as exc:  # noqa: BLE001
        return False, f"{exc.__class__.__name__}: {exc}"


def _check_overwrite(obj: Any, *, globals_ns: dict, name_hint: Optional[str]) -> bool:
    """
    Check if injecting would overwrite existing variables and prompt user.

    Returns:
        True if should proceed, False if user cancelled
    """
    # Determine what names would be used
    names = []
    if name_hint:
        names.append(name_hint)

    obj_twin_name = getattr(obj, "name", None)
    if obj_twin_name and obj_twin_name not in names:
        names.append(obj_twin_name)

    obj_name = getattr(obj, "__name__", None)
    if obj_name:
        names.append(obj_name)

    if not names:
        names.append(type(obj).__name__)

    unique_names = list(dict.fromkeys(names))

    # Check which names already exist
    existing = {}
    for name in unique_names:
        if name in globals_ns:
            existing[name] = globals_ns[name]

    if not existing:
        # Nothing to overwrite
        return True

    # Show comparison
    print("⚠️  WARNING: The following variables will be overwritten:")
    print()

    for name, current_obj in existing.items():
        # Show current value
        current_type = type(current_obj).__name__
        current_repr = repr(current_obj)
        if len(current_repr) > 60:
            current_repr = current_repr[:57] + "..."

        print(f"  Variable: {name}")
        print(f"    Current:  {current_type} = {current_repr}")

        # Show new value
        new_type = type(obj).__name__
        new_repr = repr(obj)
        if len(new_repr) > 60:
            new_repr = new_repr[:57] + "..."
        print(f"    New:      {new_type} = {new_repr}")
        print()

    # Prompt for confirmation
    try:
        response = input("Overwrite? [y/N]: ").strip().lower()
        return response in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def _inject(obj: Any, *, globals_ns: dict, name_hint: Optional[str]) -> list[str]:
    """
    Bind deserialized object into the provided namespace.

    Returns:
        List of names used to inject the object
    """
    if isinstance(obj, dict):
        globals_ns.update(obj)
        return list(obj.keys())

    names = []
    if name_hint:
        names.append(name_hint)

    # For Twin objects, also try the Twin's .name attribute
    obj_twin_name = getattr(obj, "name", None)
    if obj_twin_name and obj_twin_name not in names:
        names.append(obj_twin_name)

    obj_name = getattr(obj, "__name__", None)
    if obj_name:
        names.append(obj_name)

    # Fallback to the type name if we have nothing else
    if not names:
        names.append(type(obj).__name__)

    unique_names = list(dict.fromkeys(names))  # preserve order, drop duplicates
    for n in unique_names:
        globals_ns[n] = obj

    return unique_names


def _next_file(inbox: Path | str) -> Optional[Path]:
    inbox_path = Path(inbox)
    candidates = sorted(inbox_path.glob("*.beaver"))
    return candidates[0] if candidates else None


def find_by_id(inbox: Path | str, envelope_id: str) -> Optional[BeaverEnvelope]:
    """Locate and load a .beaver file by envelope_id."""
    inbox_path = Path(inbox)
    for path in sorted(inbox_path.glob("*.beaver")):
        with contextlib.suppress(Exception):
            record = json.loads(path.read_text())
            if record.get("envelope_id") == envelope_id:
                return read_envelope(path)
    return None


def load_by_id(
    inbox: Path | str,
    envelope_id: str,
    *,
    inject: bool = True,
    globals_ns: Optional[dict] = None,
    strict: bool = False,
    policy=None,
) -> Tuple[Optional[BeaverEnvelope], Optional[Any]]:
    """Load and inject an envelope payload by id into caller's globals."""
    env = find_by_id(inbox, envelope_id)
    if env is None:
        return None, None
    obj = unpack(env, strict=strict, policy=policy)
    if inject:
        if globals_ns is None:
            frame = inspect.currentframe()
            globals_ns = frame.f_back.f_globals if frame and frame.f_back else globals()
        _inject(obj, globals_ns=globals_ns, name_hint=env.name)
    return env, obj


def list_inbox(inbox: Path | str) -> list[BeaverEnvelope]:
    """List and load all .beaver envelopes in an inbox."""
    inbox_path = Path(inbox)
    envelopes = []
    for p in sorted(inbox_path.glob("*.beaver")):
        with contextlib.suppress(Exception):
            envelopes.append(read_envelope(p))
    return envelopes


class InboxView:
    """Wrapper to pretty-print an inbox as a table in notebooks."""

    def __init__(self, inbox: Path | str, envelopes: list[BeaverEnvelope]) -> None:
        self.inbox = Path(inbox)
        self.envelopes = envelopes

    def __len__(self) -> int:
        return len(self.envelopes)

    def __getitem__(self, key):
        """Index by position (int), envelope_id (str), or name (str)."""
        if isinstance(key, int):
            return self.envelopes[key]
        elif isinstance(key, str):
            # Try matching by envelope_id first
            for env in self.envelopes:
                if env.envelope_id == key or env.envelope_id.startswith(key):
                    return env
            # Try matching by name
            for env in self.envelopes:
                if env.name == key:
                    return env
            raise KeyError(f"No envelope with id or name matching: {key}")
        elif isinstance(key, slice):
            return self.envelopes[key]
        else:
            raise TypeError(f"Invalid index type: {type(key)}")

    def _rows(self):
        return [
            {
                "name": e.name or "(unnamed)",
                "id": e.envelope_id[:12] + "...",
                "sender": e.sender,
                "type": e.manifest.get("type"),
                "size_bytes": e.manifest.get("size_bytes"),
                "created_at": e.created_at[:19].replace("T", " "),
                "reply_to": e.reply_to[:12] + "..." if e.reply_to else "",
            }
            for e in self.envelopes
        ]

    def __repr__(self) -> str:
        rows = self._rows()
        if not rows:
            return f"InboxView({self.inbox}): empty"
        # simple text table
        headers = ["name", "id", "sender", "type", "size_bytes", "created_at", "reply_to"]
        lines = [" | ".join(headers)]
        for r in rows:
            lines.append(" | ".join(str(r.get(h, "")) for h in headers))
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        rows = self._rows()
        if not rows:
            return f"<b>InboxView({self.inbox}): empty</b>"
        headers = ["name", "id", "sender", "type", "size_bytes", "created_at", "reply_to"]
        html = ["<table>", "<thead><tr>"]
        html += [f"<th>{h}</th>" for h in headers]
        html += ["</tr></thead><tbody>"]
        for r in rows:
            html.append("<tr>")
            for h in headers:
                html.append(f"<td>{r.get(h, '')}</td>")
            html.append("</tr>")
        html.append("</tbody></table>")
        return "".join(html)


def _filtered_namespace(ns: dict, include_private: bool) -> dict:
    filtered = {}
    for k, v in ns.items():
        if not include_private and k.startswith("_"):
            continue
        if k == "__builtins__":
            continue
        if isinstance(v, (types.ModuleType, types.FrameType)):
            continue
        if isinstance(v, re.Pattern):
            continue
        filtered[k] = v
    return filtered


def _serializable_subset(payload: dict, fory: pyfory.Fory) -> dict:
    """Return only items that can be serialized by the given fory instance."""
    serializable = {}
    for k, v in payload.items():
        try:
            fory.dumps(v)
            serializable[k] = v
        except Exception:
            continue
    return serializable


def save(
    path: Path | str,
    globals_ns: Optional[dict] = None,
    *,
    include_private: bool = False,
    strict: bool = False,
    policy=None,
) -> Path:
    """Capture a namespace to a specific .beaver file path."""
    ns = globals_ns if globals_ns is not None else globals()
    payload = _filtered_namespace(ns, include_private=include_private)
    fory = pyfory.Fory(xlang=False, ref=True, strict=strict, policy=policy)
    payload = _serializable_subset(payload, fory)
    env = pack(
        payload,
        sender="snapshot",
        name="namespace",
        strict=strict,
        policy=policy,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_envelope_record(env), indent=2))
    return path


def load(
    path: Path | str,
    globals_ns: Optional[dict] = None,
    *,
    inject: bool = True,
    strict: bool = False,
    policy=None,
) -> Any:
    """Load a .beaver file and inject into caller's namespace."""
    env = read_envelope(path)
    _install_builtin_aliases()
    obj = unpack(env, strict=strict, policy=policy)
    if inject:
        if globals_ns is None:
            frame = inspect.currentframe()
            globals_ns = frame.f_back.f_globals if frame and frame.f_back else globals()
        _inject(obj, globals_ns=globals_ns, name_hint=env.name)
    return obj


def snapshot(
    out_dir: Path | str = ".",
    globals_ns: Optional[dict] = None,
    *,
    include_private: bool = False,
    strict: bool = False,
    policy=None,
) -> Path:
    """Capture the current namespace to an auto-named .beaver file."""
    ns = globals_ns if globals_ns is not None else globals()
    payload = _filtered_namespace(ns, include_private=include_private)
    fory = pyfory.Fory(xlang=False, ref=True, strict=strict, policy=policy)
    payload = _serializable_subset(payload, fory)
    env = pack(
        payload,
        sender="snapshot",
        name="namespace",
        strict=strict,
        policy=policy,
    )
    return write_envelope(env, out_dir=out_dir)


def _install_builtin_aliases() -> None:
    """
    Install common collections.abc aliases into builtins for backward compatibility.

    Older dumps may reference global names like `Iterable` that are not in builtins.
    """
    import builtins
    import typing

    aliases = {
        "Iterable": collections.abc.Iterable,
        "Mapping": collections.abc.Mapping,
        "MutableMapping": collections.abc.MutableMapping,
        "Sequence": collections.abc.Sequence,
        "Set": collections.abc.Set,
        "Optional": typing.Optional,
        "Union": typing.Union,
        "Any": typing.Any,
        "Tuple": typing.Tuple,
        "List": typing.List,
        "Dict": typing.Dict,
    }
    for name, obj in aliases.items():
        if not hasattr(builtins, name):
            setattr(builtins, name, obj)


def listen_once(
    *,
    inbox: Path | str,
    outbox: Optional[Path | str] = None,
    inject_globals: bool = False,
    autorun: bool = False,
    globals_ns: Optional[dict] = None,
    strict: bool = False,
    policy=None,
    delete_after: bool = False,
) -> Tuple[Optional[BeaverEnvelope], Optional[Any], Optional[Any]]:
    """
    Process the oldest .beaver file in inbox once.

    Returns (envelope, obj, result). If no file exists, all are None.
    """
    path = _next_file(inbox)
    if path is None:
        return None, None, None

    envelope = read_envelope(path)
    obj = unpack(envelope, strict=strict, policy=policy)

    target_globals = globals_ns if globals_ns is not None else globals()
    if inject_globals:
        _inject(obj, globals_ns=target_globals, name_hint=envelope.name)

    result = None
    if autorun and callable(obj):
        result = obj()

    if outbox is not None:
        reply_env = pack(
            result,
            sender="listener",
            name=f"{envelope.name}_result" if envelope.name else "result",
            reply_to=envelope.envelope_id,
            strict=strict,
            policy=policy,
        )
        write_envelope(reply_env, out_dir=outbox)

    if delete_after:
        path.unlink()

    return envelope, obj, result


@dataclass
class SendResult:
    path: Path
    envelope: BeaverEnvelope

    @property
    def envelope_id(self) -> str:
        return self.envelope.envelope_id

    def __repr__(self) -> str:
        """Beautiful SendResult display."""
        lines = []
        lines.append("━" * 70)
        lines.append("📤 Send Result")
        lines.append("━" * 70)

        # Extract recipient from path (e.g., shared/bob/file.beaver -> bob)
        recipient = "unknown"
        path_parts = self.path.parts
        if len(path_parts) >= 2:
            recipient = path_parts[-2]  # The directory before the filename

        # Envelope info
        lines.append(f"📧 Envelope ID: \033[36m{self.envelope.envelope_id[:16]}...\033[0m")
        lines.append(f"📝 Name: \033[36m{self.envelope.name}\033[0m")
        lines.append(f"👤 Sender: {self.envelope.sender} → \033[32mRecipient: {recipient}\033[0m")

        # File path
        lines.append(f"📁 File: \033[35m{self.path}\033[0m")

        # Content type from manifest
        obj_type = self.envelope.manifest.get("type", "Unknown")
        envelope_type = self.envelope.manifest.get("envelope_type", "")
        type_display = f"{obj_type} ({envelope_type})" if envelope_type else obj_type
        lines.append(f"📦 Type: {type_display}")

        # Status
        lines.append("")
        lines.append(f"✅ Successfully sent to {recipient}'s inbox")
        lines.append("━" * 70)

        return "\n".join(lines)


def export(
    *,
    sender: str = "unknown",
    out_dir: Path | str = "outbox",
    name: Optional[str] = None,
    inputs: Optional[Iterable[str]] = None,
    outputs: Optional[Iterable[str]] = None,
    requirements: Optional[Iterable[str]] = None,
    strict: bool = False,
    policy=None,
    context=None,
):
    """
    Decorator that adds .bind() and .send() to a callable.
    """

    def decorator(func):
        # Capture decorator's out_dir for use in BoundFunction
        default_out_dir = out_dir

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check if any arguments are RemoteVarPointers or Twins
            from uuid import uuid4

            from .computation import ComputationRequest, RemoteComputationPointer
            from .remote_vars import RemoteVarPointer
            from .twin import Twin

            has_remote_pointer = False
            has_twin = False
            destination_user = None
            twin_args = []

            # Check args for RemoteVarPointers and Twins
            for arg in args:
                if isinstance(arg, RemoteVarPointer):
                    has_remote_pointer = True
                    destination_user = arg.remote_var.owner
                    break
                elif isinstance(arg, Twin):
                    has_twin = True
                    twin_args.append(arg)
                    if not destination_user:
                        destination_user = arg.owner

            # Check kwargs
            if not has_remote_pointer and not has_twin:
                for arg in kwargs.values():
                    if isinstance(arg, RemoteVarPointer):
                        has_remote_pointer = True
                        destination_user = arg.remote_var.owner
                        break
                    elif isinstance(arg, Twin):
                        has_twin = True
                        twin_args.append(arg)
                        if not destination_user:
                            destination_user = arg.owner

            # If remote pointers detected, return computation pointer (legacy behavior)
            if has_remote_pointer:
                comp = RemoteComputationPointer(
                    func=func,
                    args=args,
                    kwargs=kwargs,
                    destination=destination_user or "unknown",
                    result_name=name or func.__name__,
                    context=context,
                )
                return comp

            # If Twins detected, return Twin result
            if has_twin:
                # Execute on public data immediately with output capture
                from .remote_vars import _ensure_sparse_shapes, _SafeDisplayProxy

                def unwrap_for_computation(val):
                    """Unwrap display proxies and ensure sparse shapes."""
                    if isinstance(val, _SafeDisplayProxy):
                        val = val._raw
                    return _ensure_sparse_shapes(val)

                public_args = tuple(
                    unwrap_for_computation(arg.public) if isinstance(arg, Twin) else arg
                    for arg in args
                )
                public_kwargs = {
                    k: unwrap_for_computation(v.public) if isinstance(v, Twin) else v
                    for k, v in kwargs.items()
                }

                # Capture stdout, stderr, and matplotlib figures
                import io
                from contextlib import redirect_stderr, redirect_stdout

                stdout_capture = io.StringIO()
                stderr_capture = io.StringIO()
                captured_figures = []

                # Try to capture matplotlib figures
                try:
                    import matplotlib
                    import matplotlib.pyplot as plt

                    from .twin import CapturedFigure

                    # Store existing figure numbers to know which are new
                    existing_figs = set(plt.get_fignums())

                    # Use non-interactive backend temporarily to prevent auto-display
                    original_backend = matplotlib.get_backend()
                    with contextlib.suppress(Exception):
                        # Agg is a non-interactive backend that won't display
                        matplotlib.use("Agg", force=True)

                    has_matplotlib = True

                    # Hook plt.show() to capture figures BEFORE they're closed
                    original_show = plt.show

                    def capturing_show(*_args, **_kwargs):
                        """Capture all current figures when show() is called."""
                        nonlocal captured_figures
                        for fig_num in plt.get_fignums():
                            if fig_num not in existing_figs:
                                fig = plt.figure(fig_num)
                                buf = io.BytesIO()
                                try:
                                    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
                                    buf.seek(0)
                                    captured_figures.append(
                                        CapturedFigure(
                                            {
                                                "figure": None,
                                                "png_bytes": buf.getvalue(),
                                            }
                                        )
                                    )
                                except Exception:
                                    pass
                        # Don't call original show - we're in Agg backend

                    plt.show = capturing_show

                except ImportError:
                    has_matplotlib = False
                    existing_figs = set()

                try:
                    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                        public_result = func(*public_args, **public_kwargs)

                    # Capture any remaining matplotlib figures and restore show
                    if has_matplotlib:
                        from .twin import CapturedFigure

                        plt.show = original_show

                        # Collect figures to capture: new figures AND figures from returned Axes
                        figures_to_capture = set()

                        # Add any new figures
                        new_figs = set(plt.get_fignums()) - existing_figs
                        for fig_num in new_figs:
                            figures_to_capture.add(plt.figure(fig_num))

                        # Also check if result contains Axes - if so, capture their figures
                        def extract_axes_figures(obj):
                            """Extract parent figures from Axes objects in result."""
                            figs = set()
                            try:
                                from matplotlib.axes import Axes

                                if isinstance(obj, Axes):
                                    if obj.figure and obj.figure.number not in existing_figs:
                                        figs.add(obj.figure)
                                elif isinstance(obj, (list, tuple)):
                                    for item in obj:
                                        figs.update(extract_axes_figures(item))
                                elif isinstance(obj, dict):
                                    for item in obj.values():
                                        figs.update(extract_axes_figures(item))
                            except Exception:
                                pass
                            return figs

                        figures_to_capture.update(extract_axes_figures(public_result))

                        # Capture all unique figures
                        captured_fig_ids = set()
                        for fig in figures_to_capture:
                            if id(fig) in captured_fig_ids:
                                continue
                            captured_fig_ids.add(id(fig))
                            buf = io.BytesIO()
                            try:
                                fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
                                buf.seek(0)
                                captured_figures.append(
                                    CapturedFigure(
                                        {
                                            "figure": fig,
                                            "png_bytes": buf.getvalue(),
                                        }
                                    )
                                )
                            except Exception:
                                pass
                            plt.close(fig)

                        plt.close("all")
                        with contextlib.suppress(Exception):
                            matplotlib.use(original_backend, force=True)

                    public_stdout = stdout_capture.getvalue()
                    public_stderr = stderr_capture.getvalue()

                except Exception as e:
                    public_result = None
                    public_stdout = stdout_capture.getvalue()
                    public_stderr = stderr_capture.getvalue() + f"\n{e}"
                    print(f"⚠️  Public execution failed: {e}")
                    # Restore backend and show on error too
                    if has_matplotlib:
                        plt.show = original_show
                        plt.close("all")
                        with contextlib.suppress(Exception):
                            matplotlib.use(original_backend, force=True)

                # Create ComputationRequest for private execution
                comp_id = uuid4().hex
                comp_request = ComputationRequest(
                    comp_id=comp_id,
                    result_id=uuid4().hex,
                    func=func,
                    args=args,
                    kwargs=kwargs,
                    sender=context.user if context else "unknown",
                    result_name=name or f"{func.__name__}_result",
                )

                # Handle None results - use sentinel dict to allow proper display
                # Functions that return None (like plotting functions) are valid
                public_value = public_result
                if public_value is None:
                    public_value = {"_none_result": True, "has_figures": len(captured_figures) > 0}

                # Return Twin with public result and computation request
                result_twin = Twin(
                    public=public_value,
                    private=comp_request,  # Computation request, not result yet
                    owner=destination_user or "unknown",
                    name=name or f"{func.__name__}_result",
                )

                # Attach captured outputs to the Twin
                result_twin.public_stdout = public_stdout
                result_twin.public_stderr = public_stderr
                result_twin.public_figures = captured_figures
                # Private outputs will be populated when .request_private() executes
                result_twin.private_stdout = None
                result_twin.private_stderr = None
                result_twin.private_figures = None

                return result_twin

            # Normal execution if no remote pointers or twins
            return func(*args, **kwargs)

        class BoundFunction:
            """A function with bound arguments ready to send."""

            def __init__(self, bound_func, bound_args, bound_kwargs):
                self.bound_func = bound_func
                self.bound_args = bound_args
                self.bound_kwargs = bound_kwargs

            def send(self, *, user=None, out_dir=None):
                """Send the bound function to a destination."""
                partial = functools.partial(self.bound_func, *self.bound_args, **self.bound_kwargs)

                # Determine destination directory
                if out_dir is not None:
                    dest_dir = out_dir
                elif user is not None:
                    dest_dir = Path("shared") / user
                else:
                    dest_dir = default_out_dir

                env = pack(
                    partial,
                    sender=sender,
                    name=name or self.bound_func.__name__,
                    inputs=list(inputs or []),
                    outputs=list(outputs or []),
                    requirements=list(requirements or []),
                    strict=strict,
                    policy=policy,
                )
                path = write_envelope(env, out_dir=dest_dir)
                return SendResult(path=path, envelope=env)

        def bind(*args, **kwargs):
            """Bind arguments to the function for later sending."""
            return BoundFunction(func, args, kwargs)

        def send(*, user=None, out_dir=None):
            """Send the function (unbound) to a destination."""
            return bind().send(user=user, out_dir=out_dir)

        wrapper.bind = bind  # type: ignore[attr-defined]
        wrapper.send = send  # type: ignore[attr-defined]
        return wrapper

    return decorator


def wait_for_reply(
    *,
    inbox: Path | str,
    reply_to: str,
    timeout: float = 30.0,
    poll_interval: float = 0.5,
    strict: bool = False,
    policy=None,
    delete_after: bool = False,
) -> Tuple[Optional[BeaverEnvelope], Optional[Any]]:
    """
    Poll an inbox for a reply envelope whose reply_to matches.
    """
    inbox_path = Path(inbox)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for path in sorted(inbox_path.glob("*.beaver")):
            try:
                record = json.loads(path.read_text())
            except Exception:
                continue
            if record.get("reply_to") not in {reply_to, Path(reply_to).name}:
                continue
            env = read_envelope(path)
            obj = unpack(env, strict=strict, policy=policy)
            if delete_after:
                path.unlink()
            return env, obj
        time.sleep(poll_interval)
    return None, None


class UserRemoteVars:
    """Helper for accessing another user's remote variables."""

    def __init__(self, username: str, context):
        self.username = username
        self.context = context
        self._remote_vars_view = None

    @property
    def remote_vars(self):
        """
        Access the remote user's published variables.

        Returns:
            RemoteVarView for browsing their vars
        """
        if self._remote_vars_view is None:
            from .remote_vars import RemoteVarView

            registry_path = self.context._public_dir / self.username / "remote_vars.json"
            data_dir = self.context._base_dir / self.username

            self._remote_vars_view = RemoteVarView(
                remote_user=self.username,
                local_user=self.context.user,
                registry_path=registry_path,
                data_dir=data_dir,
                context=self.context,
            )
        return self._remote_vars_view


class BeaverContext:
    """Connection helper holding identity and inbox/outbox defaults."""

    def __init__(
        self,
        *,
        inbox: Path | str,
        outbox: Optional[Path | str] = None,
        user: str = "unknown",
        biovault: Optional[str] = None,
        strict: bool = False,
        policy=None,
        auto_load_replies: bool = True,
        poll_interval: float = 2.0,
    ) -> None:
        self.inbox_path = Path(inbox)
        self.outbox = Path(outbox) if outbox is not None else self.inbox_path
        self.user = user
        self.biovault = biovault
        self.strict = strict
        self.policy = policy

        # Settings
        from .settings import BeaverSettings

        self._settings = BeaverSettings()

        # Remote vars setup
        self._base_dir = self.inbox_path.parent.parent  # shared/
        self._public_dir = self._base_dir / "shared" / "public"
        self._registry_path = self._public_dir / self.user / "remote_vars.json"
        self._remote_vars_registry = None

        # Staging area for remote computations
        self._staging_area = None

        # Auto-load replies setup
        self._auto_load_enabled = False
        self._poll_interval = poll_interval
        self._auto_load_thread = None
        self._stop_auto_load = False
        self._processed_replies = set()  # Track processed envelope IDs

        # Start auto-loading if enabled
        if auto_load_replies:
            self.start_auto_load()

    @property
    def settings(self):
        """
        Access configuration settings.

        Returns:
            BeaverSettings object for managing configuration
        """
        return self._settings

    @property
    def remote_vars(self):
        """
        Access this user's published remote variables registry.

        Returns:
            RemoteVarRegistry for managing published vars
        """
        if self._remote_vars_registry is None:
            from .remote_vars import RemoteVarRegistry

            self._remote_vars_registry = RemoteVarRegistry(
                owner=self.user, registry_path=self._registry_path
            )
        return self._remote_vars_registry

    def peer(self, username: str):
        """
        Access another user's published remote variables.

        Args:
            username: The peer user whose vars to view

        Returns:
            UserRemoteVars helper object
        """
        return UserRemoteVars(username=username, context=self)

    def workspace(self, **kwargs):
        """
        Get unified workspace view of shared data.

        Shows all data/actions shared between users with status and history.

        Args:
            **kwargs: Filter options (user, status, item_type)

        Returns:
            WorkspaceView with filtering options

        Examples:
            bv.workspace()                    # All items
            bv.workspace(user="bob")          # Only bob's items
            bv.workspace(status="live")       # Only live items
            bv.workspace(item_type="Twin")    # Only Twins
        """
        from .workspace import WorkspaceView

        ws = WorkspaceView(self)
        if kwargs:
            return ws(**kwargs)
        return ws

    @property
    def staged(self):
        """
        Access the staging area for remote computations.

        Returns:
            StagingArea for managing staged computations
        """
        if self._staging_area is None:
            from .computation import StagingArea

            self._staging_area = StagingArea(context=self)
        return self._staging_area

    def _add_staged(self, computation):
        """Add a computation to the staging area (internal use)."""
        self.staged.add(computation)

    def send_staged(self):
        """Send all staged computations."""
        self.staged.send_all()

    def start_auto_load(self):
        """Start auto-loading replies in background."""
        if self._auto_load_enabled:
            return  # Already running

        import threading

        self._auto_load_enabled = True
        self._stop_auto_load = False

        def auto_load_loop():
            """Background thread that polls for new replies."""
            import time

            while not self._stop_auto_load:
                with contextlib.suppress(Exception):
                    self._check_and_load_replies()

                time.sleep(self._poll_interval)

        self._auto_load_thread = threading.Thread(
            target=auto_load_loop, daemon=True, name=f"beaver-autoload-{self.user}"
        )
        self._auto_load_thread.start()
        print(
            f"🔄 Auto-load replies enabled for {self.user} (polling every {self._poll_interval}s)"
        )

    def stop_auto_load(self):
        """Stop auto-loading replies."""
        if not self._auto_load_enabled:
            return

        self._stop_auto_load = True
        self._auto_load_enabled = False
        print(f"⏸️  Auto-load replies disabled for {self.user}")

    def _check_and_load_replies(self):
        """Check inbox for new replies and auto-load them."""
        # Get all envelopes
        envelopes = list_inbox(self.inbox_path)

        # Find replies we haven't processed yet
        for envelope in envelopes:
            # Skip if already processed
            if envelope.envelope_id in self._processed_replies:
                continue

            # Check if this is a reply
            if envelope.reply_to:
                # Mark as processed first to avoid duplicates
                self._processed_replies.add(envelope.envelope_id)

                try:
                    # Load the reply
                    obj = unpack(envelope, strict=self.strict, policy=self.policy)

                    # First, try to update the original computation pointer
                    from .computation import _update_computation_pointer, _update_twin_result

                    updated = _update_computation_pointer(envelope.reply_to, obj)

                    if updated:
                        print("✨ Auto-updated computation pointer!")
                        print("   Variable holding the pointer now has result")
                        print("   Access with .value or just print the variable")

                    # Also try to update Twin results
                    twin_updated = _update_twin_result(envelope.reply_to, obj)

                    if not updated and not twin_updated:
                        # If no pointer found, inject into globals as before
                        # Get caller's globals (walk up to find __main__ or first non-beaver module)
                        target_globals = None
                        frame = inspect.currentframe()
                        while frame:
                            frame_globals = frame.f_globals
                            module_name = frame_globals.get("__name__", "")
                            if module_name == "__main__" or not module_name.startswith("beaver"):
                                target_globals = frame_globals
                                break
                            frame = frame.f_back

                        if target_globals is None:
                            # Fallback to main module
                            import __main__

                            target_globals = __main__.__dict__

                        # Inject into globals with envelope name
                        var_name = envelope.name or f"reply_{envelope.envelope_id[:8]}"
                        target_globals[var_name] = obj

                        print(f"✨ Auto-loaded reply: {var_name} = {type(obj).__name__}")
                        print(f"   From: {envelope.sender}")
                        print(f"   Reply to: {envelope.reply_to[:8]}...")

                except Exception as e:
                    # Don't fail the whole loop on one bad envelope
                    print(f"⚠️  Failed to auto-load {envelope.name}: {e}")

    def __call__(self, func=None, **kwargs):
        """Allow @bv as a decorator; kwargs override context defaults."""

        def decorator(f):
            # capture base paths for user targeting inside the wrapper
            base_out = Path(kwargs.get("out_dir", self.outbox))
            base_parent = base_out.parent

            def wrapped_export(func):
                wrapped = export(
                    sender=kwargs.get("sender", self.user),
                    out_dir=base_out,
                    name=kwargs.get("name"),
                    inputs=kwargs.get("inputs"),
                    outputs=kwargs.get("outputs"),
                    requirements=kwargs.get("requirements"),
                    strict=kwargs.get("strict", self.strict),
                    policy=kwargs.get("policy", self.policy),
                    context=self,
                )(func)

                original_send = wrapped.send

                def send_with_user(*, user=None, out_dir=None):
                    dest_dir = out_dir
                    if dest_dir is None and user:
                        dest_dir = base_parent / user
                    return original_send(user=user, out_dir=dest_dir)

                wrapped.send = send_with_user  # type: ignore[attr-defined]
                return wrapped

            return wrapped_export(f)

        if func is None:
            return decorator
        return decorator(func)

    def send(self, obj: Any, **kwargs) -> Path:
        """Pack and write an object using context defaults."""
        # Priority for name:
        # 1. Explicit name passed to send()
        # 2. Object's .name attribute (e.g., Twin.name)
        # 3. Auto-detected variable name
        # 4. None
        name = kwargs.get("name")
        if name is None:
            # Check if object has a .name attribute (like Twin)
            if hasattr(obj, "name") and obj.name:
                name = obj.name
            else:
                # Auto-detect from caller's variable name
                name = _get_var_name_from_caller(obj, depth=2)

        env = pack(
            obj,
            sender=kwargs.get("sender", self.user),
            name=name,
            inputs=kwargs.get("inputs"),
            outputs=kwargs.get("outputs"),
            requirements=kwargs.get("requirements"),
            reply_to=kwargs.get("reply_to"),
            strict=kwargs.get("strict", self.strict),
            policy=kwargs.get("policy", self.policy),
            preserve_private=kwargs.get("preserve_private", False),
        )
        out_dir = kwargs.get("out_dir")
        to_user = kwargs.get("user")
        if out_dir is None:
            if to_user:
                base = Path(self.outbox).parent
                out_dir = base / to_user
            else:
                out_dir = self.outbox
        path = write_envelope(env, out_dir=out_dir)
        return SendResult(path=path, envelope=env)

    def reply(self, obj: Any, *, to_id: str, **kwargs) -> Path:
        """Send a reply payload correlated to a prior envelope id."""
        return self.send(obj, reply_to=to_id, **kwargs)

    def listen_once(self, **kwargs):
        """Listen with context inbox/outbox defaults."""
        return listen_once(
            inbox=kwargs.get("inbox", self.inbox_path),
            outbox=kwargs.get("outbox", self.outbox),
            inject_globals=kwargs.get("inject_globals", False),
            autorun=kwargs.get("autorun", False),
            globals_ns=kwargs.get("globals_ns"),
            strict=kwargs.get("strict", self.strict),
            policy=kwargs.get("policy", self.policy),
            delete_after=kwargs.get("delete_after", False),
        )

    def wait_for_reply(self, reply_to: str, **kwargs):
        """Wait for a reply matching reply_to in the context inbox."""
        return wait_for_reply(
            inbox=kwargs.get("inbox", self.inbox_path),
            reply_to=reply_to,
            timeout=kwargs.get("timeout", 30.0),
            poll_interval=kwargs.get("poll_interval", 0.5),
            strict=kwargs.get("strict", self.strict),
            policy=kwargs.get("policy", self.policy),
            delete_after=kwargs.get("delete_after", False),
        )

    def wait_for_message(
        self,
        *,
        timeout: float = 60.0,
        poll_interval: float = 1.0,
        filter_sender: Optional[str] = None,
        filter_name: Optional[str] = None,
        auto_load: bool = True,
    ):
        """
        Wait for a new message to arrive in the inbox.

        Args:
            timeout: Max seconds to wait (default 60)
            poll_interval: Seconds between checks (default 1)
            filter_sender: Only wait for messages from this sender
            filter_name: Only wait for messages with this name
            auto_load: If True, automatically load and return the object

        Returns:
            If auto_load: (envelope, loaded_object)
            Otherwise: envelope

        Example:
            # Wait for any message
            env, obj = bv.wait_for_message()

            # Wait for message from specific sender
            env, obj = bv.wait_for_message(filter_sender="alice")

            # Wait with custom timeout
            env, obj = bv.wait_for_message(timeout=120)
        """
        # Get current envelope IDs to detect new ones
        current_ids = {env.envelope_id for env in list_inbox(self.inbox_path)}

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            # Check for new envelopes
            for env in list_inbox(self.inbox_path):
                if env.envelope_id in current_ids:
                    continue  # Not new

                # Apply filters
                if filter_sender and env.sender != filter_sender:
                    continue
                if filter_name and env.name != filter_name:
                    continue

                # Found a new matching envelope!
                print(f"📬 New message: {env.name or env.envelope_id[:12]}")
                print(f"   From: {env.sender}")

                if auto_load:
                    obj = unpack(env, strict=self.strict, policy=self.policy)
                    return env, obj
                return env

            time.sleep(poll_interval)

        print(f"⏰ Timeout after {timeout}s waiting for message")
        return None, None if auto_load else None

    def load_by_id(self, envelope_id: str, **kwargs):
        """Load a payload by id using context inbox and inject into caller's globals."""
        inject = kwargs.get("inject", True)
        globals_ns = kwargs.get("globals_ns")

        # Auto-detect caller's globals if injecting and not provided
        if inject and globals_ns is None:
            frame = inspect.currentframe()
            if frame and frame.f_back:
                globals_ns = frame.f_back.f_globals

        return load_by_id(
            inbox=kwargs.get("inbox", self.inbox_path),
            envelope_id=envelope_id,
            inject=inject,
            globals_ns=globals_ns,
            strict=kwargs.get("strict", self.strict),
            policy=kwargs.get("policy", self.policy),
        )

    def list_inbox(self, **kwargs):
        """List envelopes in the context inbox."""
        return list_inbox(kwargs.get("inbox", self.inbox_path))

    def inbox(
        self,
        sort_by: Optional[str] = "created_at",
        reverse: bool = False,
        newest: bool = False,
        oldest: bool = False,
        by_name: bool = False,
        by_sender: bool = False,
        by_size: bool = False,
        by_type: bool = False,
        **kwargs,
    ):
        """
        Return an InboxView for the current inbox with optional sorting.

        Args:
            sort_by: Field to sort by (default: "created_at")
            reverse: Reverse sort order (default: False for oldest first)
            newest: Shorthand for newest first (created_at desc)
            oldest: Shorthand for oldest first (created_at asc)
            by_name: Sort by name alphabetically
            by_sender: Sort by sender alphabetically
            by_size: Sort by size (largest first)
            by_type: Sort by type alphabetically

        Returns:
            InboxView with sorted envelopes
        """
        inbox_path = kwargs.get("inbox", self.inbox_path)
        envelopes = list_inbox(inbox_path)

        # Apply flag shortcuts
        if newest:
            sort_by = "created_at"
            reverse = True
        elif oldest:
            sort_by = "created_at"
            reverse = False
        elif by_name:
            sort_by = "name"
            reverse = False
        elif by_sender:
            sort_by = "sender"
            reverse = False
        elif by_size:
            sort_by = "size"
            reverse = True
        elif by_type:
            sort_by = "type"
            reverse = False

        # Sort
        if sort_by:

            def sort_key(env):
                if sort_by == "name":
                    return env.name or ""
                elif sort_by == "sender":
                    return env.sender
                elif sort_by == "created_at":
                    return env.created_at
                elif sort_by == "size":
                    return env.manifest.get("size_bytes", 0)
                elif sort_by == "type":
                    return env.manifest.get("type", "")
                return ""

            envelopes = sorted(envelopes, key=sort_key, reverse=reverse)

        return InboxView(inbox_path, envelopes)

    def save(self, path: Path | str, **kwargs) -> Path:
        """Save a namespace (default globals) using context defaults."""
        return save(
            path=path,
            globals_ns=kwargs.get("globals_ns"),
            include_private=kwargs.get("include_private", False),
            strict=kwargs.get("strict", self.strict),
            policy=kwargs.get("policy", self.policy),
        )

    def load(self, path: Path | str, **kwargs) -> Any:
        """Load a .beaver file and inject into caller's namespace."""
        inject = kwargs.get("inject", True)
        globals_ns = kwargs.get("globals_ns")

        # Auto-detect caller's globals if injecting and not provided
        if inject and globals_ns is None:
            frame = inspect.currentframe()
            if frame and frame.f_back:
                globals_ns = frame.f_back.f_globals

        return load(
            path=path,
            globals_ns=globals_ns,
            inject=inject,
            strict=kwargs.get("strict", self.strict),
            policy=kwargs.get("policy", self.policy),
        )

    def snapshot(self, **kwargs) -> Path:
        """Auto-name and write a namespace snapshot using context outbox."""
        return snapshot(
            out_dir=kwargs.get("out_dir", self.outbox),
            globals_ns=kwargs.get("globals_ns"),
            include_private=kwargs.get("include_private", False),
            strict=kwargs.get("strict", self.strict),
            policy=kwargs.get("policy", self.policy),
        )


def connect(
    folder: Path | str,
    *,
    user: str = "unknown",
    biovault: Optional[str] = None,
    inbox: Optional[Path | str] = None,
    outbox: Optional[Path | str] = None,
    strict: bool = False,
    policy=None,
    auto_load_replies: bool = True,
    poll_interval: float = 2.0,
) -> BeaverContext:
    """
    Create a BeaverContext with shared defaults.

    Args:
        folder: Base directory for shared files
        user: Username for this context
        biovault: Optional biovault identifier
        inbox: Inbox directory (defaults to folder/user)
        outbox: Outbox directory (defaults to inbox)
        strict: Enable strict mode for serialization
        policy: Security policy for deserialization
        auto_load_replies: Auto-load computation replies (default: True)
        poll_interval: How often to check for replies in seconds (default: 2.0)

    Returns:
        BeaverContext with auto-loading enabled
    """
    base = Path(folder)
    user_subdir = base / user if user else base
    return BeaverContext(
        inbox=inbox if inbox is not None else user_subdir,
        outbox=outbox if outbox is not None else user_subdir,
        user=user,
        biovault=biovault,
        strict=strict,
        policy=policy,
        auto_load_replies=auto_load_replies,
        poll_interval=poll_interval,
    )
