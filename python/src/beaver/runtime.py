from __future__ import annotations

import base64
import collections.abc
import contextlib
import functools
import inspect
import json
import os
import re
import tempfile
import textwrap
import time
import types
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional
from uuid import uuid4

from . import lib_support
from .envelope import BeaverEnvelope
from .policy import DEFAULT_POLICY, TRUSTED_POLICY, BeaverPolicy

try:
    import pyfory
except ImportError as e:
    raise ImportError(
        "pyfory is required for beaver serialization; pickle fallback is disabled for security."
    ) from e


class SecurityError(Exception):
    """Raised when a security policy violation is detected."""


try:  # pragma: no cover
    from pyfory.serializer import DataClassSerializer
except Exception:  # pragma: no cover
    DataClassSerializer = None  # type: ignore[assignment]


if DataClassSerializer is not None:

    class _LenientDataClassSerializer(DataClassSerializer):
        """Dataclass serializer that ignores schema-hash mismatches in non-compatible mode.

        This is only used as a targeted fallback for legacy Beaver internal types (Twin/LiveVar)
        when payloads were produced by older runtimes whose pyfory schema hash differs.
        """

        def _read_header(self, buffer):
            if not self.fory.compatible:
                # Legacy/non-compatible payloads write an int32 schema hash first.
                # We intentionally ignore it and continue reading fields in the current schema order.
                buffer.read_int32()
                return len(self._field_names)
            return super()._read_header(buffer)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Force using the Python implementation (not the JIT-generated read method),
            # so our overridden _read_header takes effect.
            self.read = lambda buffer: DataClassSerializer.read(self, buffer)  # type: ignore[method-assign]


if TYPE_CHECKING:
    from .session import Session

# Global context tracker for SyftBox-aware file operations
_CURRENT_CONTEXT = None

# Allowed modules for trusted loader imports (in trusted/non-interactive paths)
_ALLOWED_LOADER_MODULES = {
    "numpy",
    "pandas",
    "anndata",
    "pathlib",
    "io",
    "gzip",
    "json",
    "csv",
    "pyarrow",
}


def _get_current_context():
    """Get the current BeaverContext for SyftBox file operations."""
    return _CURRENT_CONTEXT


def _set_current_context(ctx):
    """Set the current BeaverContext."""
    global _CURRENT_CONTEXT
    _CURRENT_CONTEXT = ctx


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

    _handlers: dict[type, dict[str, Any]] = {}

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
    def get(cls, typ: type, *, auto_register: bool = True) -> Optional[dict[str, Any]]:
        h = cls._handlers.get(typ)
        if h and h.get("serializer") and h.get("deserializer"):
            return h

        # Auto-register builtin loaders if not found
        if auto_register and h is None:
            from . import lib_support

            if lib_support.register_by_type(typ, cls):
                # Check again after registration
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
    type_module = getattr(obj.__class__, "__module__", "") or ""
    if type_module.startswith("matplotlib."):
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
            # Always convert to list - TrustedLoader handles file-based serialization
            # for top-level numpy arrays, but nested arrays need to be serialized inline
            return obj.tolist()
    except ImportError:
        pass

    # Handle pandas
    try:
        import pandas as pd

        if isinstance(obj, pd.DataFrame):
            return {"_beaver_dataframe": True, "data": obj.to_dict(orient="list")}
        if isinstance(obj, pd.Series):
            return {"_beaver_series": True, "data": obj.to_dict(), "name": obj.name}
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


def _validate_artifact_path(path: str | Path, *, backend=None) -> Path:
    """Ensure artifact path stays within an allowlisted base directory."""
    candidate = Path(path)
    if ".." in candidate.parts:
        raise SecurityError(f"Path traversal detected for artifact: {path}")
    bases = []
    if backend and getattr(backend, "data_dir", None):
        bases.append(Path(backend.data_dir))
    bases.append(Path.cwd())
    # Project root (python/src/beaver/../../..)
    with contextlib.suppress(Exception):
        bases.append(Path(__file__).resolve().parents[3])
    bases.append(Path(tempfile.gettempdir()))
    # Detect SyftBox/BioVault root from "datasites" in path
    # This handles cross-session artifact references within the same SyftBox installation
    try:
        parts = candidate.resolve().parts
        if "datasites" in parts:
            idx = parts.index("datasites")
            syftbox_root = Path(*parts[:idx])
            if syftbox_root.exists():
                bases.append(syftbox_root)
    except Exception:
        pass

    resolved_candidate = candidate.resolve()
    for base in bases:
        resolved_base = base.resolve()
        try:
            resolved_candidate.relative_to(resolved_base)
            break
        except ValueError:
            continue
    else:
        raise SecurityError(f"Path traversal detected for artifact: {path}")

    if resolved_candidate.is_symlink():
        target = resolved_candidate.readlink().resolve()
        for base in bases:
            try:
                target.relative_to(base.resolve())
                break
            except ValueError:
                continue
        else:
            raise SecurityError(f"Symlink escape detected for artifact: {path}")
    return resolved_candidate


def _select_policy(policy=None, *, trust_loader: bool = False) -> BeaverPolicy:
    """Select effective policy considering env overrides and trust_loader flag."""
    if policy is not None:
        return policy
    # trust_loader=True implies TRUSTED_POLICY (allows functions)
    if trust_loader:
        return TRUSTED_POLICY
    trusted_env = os.getenv("BEAVER_TRUSTED_POLICY", "").lower() in {"1", "true", "yes"} or (
        os.getenv("BEAVER_TRUSTED_LOADERS", "").lower() in {"1", "true", "yes"}
    )
    if trusted_env:
        return TRUSTED_POLICY
    return DEFAULT_POLICY


def _analyze_loader_source(src: str, allowed_imports: Optional[set[str]] = None) -> None:
    """Static analysis of loader code to block dangerous patterns."""
    import ast

    dangerous_names = {"__import__", "eval", "exec", "compile", "open"}
    dangerous_modules = {"os", "subprocess", "sys", "importlib", "builtins"}
    allowed_imports = allowed_imports or _ALLOWED_LOADER_MODULES

    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = []
            if isinstance(node, ast.Import):
                mods = [alias.name.split(".")[0] for alias in node.names]
            elif node.module:
                mods = [node.module.split(".")[0]]
            for mod in mods:
                if mod in dangerous_modules:
                    raise SecurityError(f"Blocked import in trusted loader: {mod}")
                if allowed_imports and mod not in allowed_imports:
                    raise SecurityError(f"Import not allowed in trusted loader: {mod}")
        elif isinstance(node, ast.Call):
            fn = node.func
            name = None
            if isinstance(fn, ast.Name):
                name = fn.id
            elif isinstance(fn, ast.Attribute):
                name = fn.attr
            if name in dangerous_names:
                raise SecurityError(f"Blocked call in trusted loader: {name}")
            if (
                isinstance(fn, ast.Name)
                and fn.id == "__import__"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                modname = str(node.args[0].value).split(".")[0]
                if modname in dangerous_modules:
                    raise SecurityError(f"Blocked dynamic import: {modname}")


def _loader_defined_functions(src: str) -> list[str]:
    """Return top-level function names defined by loader source."""
    import ast

    try:
        tree = ast.parse(src)
    except Exception:
        return []
    return [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]


def _posixify_path(p: str | Path) -> str:
    return str(p).replace("\\", "/")


def _cross_platform_filename(path_str: str) -> str:
    """Extract filename from a path that may be Windows or Unix format.

    On POSIX systems, Path("C:\\Users\\...\\file.txt").name returns the entire
    string because backslashes aren't path separators. This function handles
    both Windows and Unix paths correctly on any platform.
    """
    if not path_str:
        return path_str
    normalized = path_str.replace("\\", "/")
    return normalized.rsplit("/", 1)[-1]


def _relative_posix_path(path: Path, base_dir: Optional[Path]) -> Optional[str]:
    if base_dir is None:
        return None
    try:
        rel = path.resolve().relative_to(base_dir.resolve())
    except Exception:
        return None
    return _posixify_path(rel)


def _trusted_loader_base_dir(artifact_dir: Optional[Path], backend) -> Optional[Path]:
    """Choose a base dir for TrustedLoader relative paths.

    We prefer storing paths relative to the *session* directory (not the session/data dir),
    so loaders can only reference artifacts within the session tree (e.g. `data/foo.bin`).
    """
    if artifact_dir is not None:
        resolved = Path(artifact_dir).resolve()
        if resolved.name == "data":
            return resolved.parent
        return resolved
    if backend is not None and getattr(backend, "data_dir", None) is not None:
        # Allow relative-to-datasites-root paths like `<email>/shared/...`.
        return (Path(backend.data_dir).resolve() / "datasites").resolve()
    return None


def _exec_restricted_loader(src: str, scope: dict) -> dict:
    """Execute loader code in a RestrictedPython sandbox."""
    try:
        from RestrictedPython import compile_restricted
        from RestrictedPython.Eval import default_guarded_getitem, default_guarded_getiter
        from RestrictedPython.Guards import (
            full_write_guard,
            guarded_iter_unpack_sequence,
            guarded_unpack_sequence,
            safe_builtins,
            safer_getattr,
        )
    except Exception as e:  # pragma: no cover - import error path
        raise SecurityError(f"RestrictedPython not available: {e}") from e

    bytecode = compile_restricted(src, filename="<trusted_loader>", mode="exec")
    restricted_globals: dict = {
        "__builtins__": safe_builtins,
        "_getattr_": safer_getattr,
        "_getitem_": default_guarded_getitem,
        "_getiter_": default_guarded_getiter,
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        "_unpack_sequence_": guarded_unpack_sequence,
        "_write_": full_write_guard,
    }
    restricted_globals.update(scope)
    exec(bytecode, restricted_globals, restricted_globals)
    return restricted_globals


def _prepare_for_sending(
    obj: Any,
    *,
    artifact_dir: Optional[Path] = None,
    name_hint: Optional[str] = None,
    preserve_private: bool = False,
    backend=None,
    recipients: Optional[list] = None,
) -> Any:
    """Prepare object for sending by handling trusted loaders and optionally stripping private data.

    Args:
        obj: Object to prepare
        artifact_dir: Directory for trusted loader artifacts
        name_hint: Name hint for artifacts
        preserve_private: If True, keep private data in Twins (for approved results)
        backend: Optional SyftBoxBackend for encrypted writes
        recipients: Optional list of recipient emails for encryption
    """
    from .syft_url import path_to_syft_url
    from .twin import Twin

    def _path_for_loader(path: Path, base_dir: Optional[Path]) -> str:
        syft_url = path_to_syft_url(path)
        if syft_url:
            return syft_url
        return _relative_posix_path(path, base_dir) or path.name

    def _write_artifact_encrypted(serializer, obj_to_write, path, backend, recipients):
        """Write artifact using TrustedLoader serializer, optionally encrypting.

        Returns metadata dict if serializer returns one, else None.
        """
        meta = None
        if backend and backend.uses_crypto and recipients:
            # Write to temp file first, then encrypt
            temp_path = Path(tempfile.gettempdir()) / f"_beaver_tmp_{uuid4().hex}.bin"
            try:
                meta = serializer(obj_to_write, temp_path)
                # Write the main .bin file
                data = temp_path.read_bytes()
                backend.storage.write_with_shadow(
                    absolute_path=str(path),
                    data=data,
                    recipients=recipients,
                    hint="trusted-loader-artifact",
                    overwrite=True,
                )
            finally:
                if temp_path.exists():
                    temp_path.unlink()
        else:
            # Direct write (no encryption)
            meta = serializer(obj_to_write, path)
        return meta

    # Trusted loader conversion (non-Twin)
    lib_support.register_builtin_loader(obj, TrustedLoader)
    tl = TrustedLoader.get(type(obj))
    if tl:
        target_dir = Path(artifact_dir) if artifact_dir else Path(tempfile.gettempdir())
        target_dir.mkdir(parents=True, exist_ok=True)
        base = name_hint or tl["name"].split(".")[-1] or "artifact"
        path = target_dir / f"{base}.bin"
        meta = _write_artifact_encrypted(tl["serializer"], obj, path, backend, recipients)
        import inspect

        src_lines = inspect.getsource(tl["deserializer"]).splitlines()
        src_clean = "\n".join(
            line
            for line in src_lines
            if not (
                line.lstrip().startswith("@TrustedLoader")
                or line.lstrip().startswith("@trusted_loader_cls")
            )
        )
        src_clean = textwrap.dedent(src_clean)
        base_dir = _trusted_loader_base_dir(artifact_dir, backend)
        # Prefer relative unix-style paths; avoid embedding OS absolute paths in payloads.
        path_for_loader = _path_for_loader(path, base_dir)
        result = {
            "_trusted_loader": True,
            "name": tl["name"],
            "path": path_for_loader,
            "deserializer_src": src_clean,
        }
        if meta:
            result["meta"] = meta  # Embed serializer metadata
        return result

    # Handle Twin objects
    if isinstance(obj, Twin):
        # Use object.__getattribute__ to bypass Twin's auto-load hook
        # This allows passing through TrustedLoader dicts directly for re-serialization
        public_obj = object.__getattribute__(obj, "public")
        private_obj = object.__getattribute__(obj, "private") if preserve_private else None

        # Register trusted loaders for Twin's public/private content
        # (the main obj registration above only handles the outer Twin)
        if public_obj is not None:
            lib_support.register_builtin_loader(public_obj, TrustedLoader)
        if private_obj is not None:
            lib_support.register_builtin_loader(private_obj, TrustedLoader)

        # Apply trusted loader to public side if available
        tl_pub = TrustedLoader.get(type(public_obj)) if public_obj is not None else None
        if tl_pub:
            target_dir = Path(artifact_dir) if artifact_dir else Path(tempfile.gettempdir())
            target_dir.mkdir(parents=True, exist_ok=True)
            base = name_hint or getattr(public_obj, "name", None) or "artifact"
            path = target_dir / f"{base}_public.bin"
            meta = _write_artifact_encrypted(
                tl_pub["serializer"], public_obj, path, backend, recipients
            )
            import inspect

            src_lines = inspect.getsource(tl_pub["deserializer"]).splitlines()
            src_clean = "\n".join(
                line
                for line in src_lines
                if not (
                    line.lstrip().startswith("@TrustedLoader")
                    or line.lstrip().startswith("@trusted_loader_cls")
                )
            )
            src_clean = textwrap.dedent(src_clean)
            base_dir = _trusted_loader_base_dir(artifact_dir, backend)
            path_for_loader = _path_for_loader(path, base_dir)
            public_obj = {
                "_trusted_loader": True,
                "name": tl_pub["name"],
                "path": path_for_loader,
                "deserializer_src": src_clean,
            }
            if meta:
                public_obj["meta"] = meta

        # Apply trusted loader to private side if preserving and available
        if preserve_private and private_obj is not None:
            tl_priv = TrustedLoader.get(type(private_obj))
            if tl_priv:
                target_dir = Path(artifact_dir) if artifact_dir else Path(tempfile.gettempdir())
                target_dir.mkdir(parents=True, exist_ok=True)
                base = name_hint or getattr(private_obj, "name", None) or "artifact"
                path = target_dir / f"{base}_private.bin"
                meta = _write_artifact_encrypted(
                    tl_priv["serializer"], private_obj, path, backend, recipients
                )
                import inspect

                src_lines = inspect.getsource(tl_priv["deserializer"]).splitlines()
                src_clean = "\n".join(
                    line
                    for line in src_lines
                    if not (
                        line.lstrip().startswith("@TrustedLoader")
                        or line.lstrip().startswith("@trusted_loader_cls")
                    )
                )
                src_clean = textwrap.dedent(src_clean)
                base_dir = _trusted_loader_base_dir(artifact_dir, backend)
                path_for_loader = _path_for_loader(path, base_dir)
                private_obj = {
                    "_trusted_loader": True,
                    "name": tl_priv["name"],
                    "path": path_for_loader,
                    "deserializer_src": src_clean,
                }
                if meta:
                    private_obj["meta"] = meta

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

        # Copy serializable captured outputs (public always, private only if preserving)
        for attr in ("public_stdout", "public_stderr"):
            if hasattr(obj, attr):
                setattr(new_twin, attr, getattr(obj, attr))
        if preserve_private:
            for attr in ("private_stdout", "private_stderr"):
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
        # Only include private figures if preserving private data
        if preserve_private and hasattr(obj, "private_figures") and obj.private_figures:
            new_twin.private_figures = convert_figures_for_sending(obj.private_figures)

        return new_twin

    # Handle collections containing Twins
    if isinstance(obj, dict):
        return {
            k: _prepare_for_sending(
                v,
                artifact_dir=artifact_dir,
                name_hint=k,
                preserve_private=preserve_private,
                backend=backend,
                recipients=recipients,
            )
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        result = [
            _prepare_for_sending(
                item,
                artifact_dir=artifact_dir,
                preserve_private=preserve_private,
                backend=backend,
                recipients=recipients,
            )
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
    backend=None,
    recipients: Optional[list] = None,
) -> BeaverEnvelope:
    """Serialize an object into a BeaverEnvelope (Python-native).

    Args:
        preserve_private: If True, include private data in Twins (for approved results)
        backend: Optional SyftBoxBackend for encrypted artifact writes
        recipients: Optional list of recipient emails for encryption
    """
    # Prepare for sending (optionally strip private data)
    obj_to_send = _prepare_for_sending(
        obj,
        artifact_dir=artifact_dir,
        name_hint=name,
        preserve_private=preserve_private,
        backend=backend,
        recipients=recipients,
    )

    compatible_env = os.getenv("BEAVER_FORY_COMPATIBLE")
    if compatible_env is None:
        # Default to compatible mode for cross-machine (SyftBox) sharing.
        fory_compatible = backend is not None
    else:
        fory_compatible = compatible_env.lower() in {"1", "true", "yes", "on"}

    field_nullable_env = os.getenv("BEAVER_FORY_FIELD_NULLABLE")
    if field_nullable_env is None:
        # Nullable-by-default is more tolerant to Optional/type-hint inference differences.
        fory_field_nullable = fory_compatible
    else:
        fory_field_nullable = field_nullable_env.lower() in {"1", "true", "yes", "on"}

    effective_policy = _select_policy(policy)
    fory = pyfory.Fory(
        xlang=False,
        ref=True,
        strict=strict,
        policy=effective_policy,
        compatible=fory_compatible,
        field_nullable=fory_field_nullable,
    )

    # Register Twin type so it can be deserialized
    from .twin import Twin

    fory.register_type(Twin)

    # Register LiveVar type
    from .live_var import LiveVar

    fory.register_type(LiveVar)

    payload = fory.dumps(obj_to_send)
    manifest = _summarize(obj)
    manifest["size_bytes"] = len(payload)
    manifest["language"] = "python"
    manifest["fory_compatible"] = fory_compatible
    manifest["fory_field_nullable"] = fory_field_nullable
    # Help debug cross-machine deserialization mismatches (pyfory schema hashes depend on runtime).
    try:  # pragma: no cover
        import platform

        manifest["pyfory_version"] = getattr(pyfory, "__version__", "unknown")
        manifest["python_version"] = platform.python_version()
    except Exception:  # pragma: no cover
        pass

    # If packing a ComputationRequest, add required_versions to manifest
    from .computation import ComputationRequest, _detect_function_imports_with_versions

    if isinstance(obj, ComputationRequest) and hasattr(obj, "func"):
        try:
            versions = _detect_function_imports_with_versions(obj.func)
            if versions:
                manifest["required_versions"] = versions
        except Exception:
            pass  # Don't fail if version detection fails

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
    path_obj = Path(path)
    record = json.loads(path_obj.read_text())
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
    # Non-serialized hint used to resolve TrustedLoader relative artifact paths.
    env._path = str(path_obj)  # type: ignore[attr-defined]
    return env


def unpack(
    envelope: BeaverEnvelope,
    *,
    strict: bool = False,
    policy=None,
    auto_accept: bool = False,
    backend=None,
    trust_loader: Optional[bool] = None,
) -> Any:
    """Deserialize the payload in a BeaverEnvelope.

    Args:
        envelope: The envelope to unpack
        strict: Strict deserialization mode
        policy: Deserialization policy
        auto_accept: If True, automatically accept trusted loaders without prompting
        backend: Optional SyftBoxBackend for reading encrypted artifact files
        trust_loader: If True, run loader in trusted mode. If None, prompt on failure.
    """
    _install_builtin_aliases()
    effective_policy: BeaverPolicy = _select_policy(policy, trust_loader=bool(trust_loader))

    # Prefer sender settings when available (supports cross-OS / cross-Python patch sharing).
    sender_manifest = envelope.manifest or {}
    sender_compatible = sender_manifest.get("fory_compatible")
    sender_field_nullable = sender_manifest.get("fory_field_nullable")

    compatible_env = os.getenv("BEAVER_FORY_COMPATIBLE")
    if compatible_env is None:
        # Default to sender preference; if missing, enable for SyftBox/cross-machine mode.
        fory_compatible = (
            bool(sender_compatible) if sender_compatible is not None else backend is not None
        )
    else:
        fory_compatible = compatible_env.lower() in {"1", "true", "yes", "on"}

    field_nullable_env = os.getenv("BEAVER_FORY_FIELD_NULLABLE")
    if field_nullable_env is None:
        fory_field_nullable = (
            bool(sender_field_nullable) if sender_field_nullable is not None else fory_compatible
        )
    else:
        fory_field_nullable = field_nullable_env.lower() in {"1", "true", "yes", "on"}

    def _loads_with(
        *, compatible: bool, field_nullable: bool, lenient_internal_hash: bool = False
    ) -> Any:
        f = pyfory.Fory(
            xlang=False,
            ref=True,
            strict=strict,
            policy=effective_policy,
            compatible=compatible,
            field_nullable=field_nullable,
        )

        # Register Beaver internal types.
        from .computation import ComputationRequest
        from .live_var import LiveVar
        from .twin import Twin

        if lenient_internal_hash and _LenientDataClassSerializer is not None:
            f.register_type(Twin, serializer=_LenientDataClassSerializer(f, Twin))
            f.register_type(LiveVar, serializer=_LenientDataClassSerializer(f, LiveVar))
            f.register_type(
                ComputationRequest,
                serializer=_LenientDataClassSerializer(f, ComputationRequest),
            )
        else:
            f.register_type(Twin)
            f.register_type(LiveVar)
            f.register_type(ComputationRequest)

        return f.loads(envelope.payload)

    # Block pickle payloads outright (magic 0x80 precedes pickle protocol)
    if envelope.payload.startswith(b"\x80"):
        raise SecurityError("Pickle payloads are blocked.")

    try:
        obj = _loads_with(compatible=fory_compatible, field_nullable=fory_field_nullable)
    except Exception as e:
        primary_exc = e
        primary_msg = str(e)
        last_exc = e
        last_msg = primary_msg

        def _looks_like_out_of_bound(msg: str) -> bool:
            return "out of bound" in msg or "read_varuint32" in msg

        def _looks_like_hash_mismatch(msg: str) -> bool:
            return "Hash " in msg and "not consistent with" in msg and "beaver." in msg

        fallback_failures: list[str] = []
        attempts: list[str] = [
            f"compatible={fory_compatible} field_nullable={fory_field_nullable} (primary)"
        ]

        # If we don't know what mode the sender used (no manifest flags) and we're in compatible
        # mode (common in SyftBox contexts), legacy non-compatible payloads can fail with
        # buffer out-of-bound errors. Retry in non-compatible mode.
        if (
            fory_compatible
            and compatible_env is None
            and sender_compatible is None
            and _looks_like_out_of_bound(last_msg)
        ):
            try:
                attempts.append("compatible=False field_nullable=False (fallback legacy)")
                obj = _loads_with(compatible=False, field_nullable=False)
            except Exception as retry_exc:
                last_exc = retry_exc
                last_msg = str(retry_exc)
                fallback_failures.append(f"legacy->non-compatible failed: {last_msg}")

        # Legacy non-compatible payloads can fail on internal-type schema hash mismatches.
        attempted_legacy_fallback = any("fallback legacy" in a for a in attempts)
        if (not fory_compatible or attempted_legacy_fallback) and _looks_like_hash_mismatch(
            last_msg
        ):
            try:
                attempts.append(
                    "compatible=False field_nullable=<same> lenient_internal_hash=True (fallback)"
                )
                obj = _loads_with(
                    compatible=False,
                    field_nullable=fory_field_nullable,
                    lenient_internal_hash=True,
                )
            except Exception as retry_exc:
                last_exc = retry_exc
                last_msg = str(retry_exc)
                fallback_failures.append(f"legacy->lenient-internal-hash failed: {last_msg}")

        if "obj" not in locals():
            msg = primary_msg
            if fallback_failures:
                msg += " (fallback failures: " + " | ".join(fallback_failures) + ")"
            if _looks_like_hash_mismatch(msg):
                sender_py = (envelope.manifest or {}).get("python_version")
                sender_pf = (envelope.manifest or {}).get("pyfory_version")
                receiver_py = None
                receiver_pf = getattr(pyfory, "__version__", None)
                try:  # pragma: no cover
                    import platform

                    receiver_py = platform.python_version()
                except Exception:  # pragma: no cover
                    receiver_py = None
                msg += (
                    " (Likely sender/receiver runtime mismatch; ensure both sides run the same Beaver code + pyfory version"
                    + (
                        f"; sender python={sender_py} pyfory={sender_pf}, receiver python={receiver_py} pyfory={receiver_pf}"
                        if (sender_py or sender_pf or receiver_py or receiver_pf)
                        else ""
                    )
                    + (
                        "; if you're sharing across OS/Python patch versions, enable compatible mode with `BEAVER_FORY_COMPATIBLE=1` and republish"
                        if not fory_compatible
                        else ""
                    )
                    + ".)"
                )
            msg += f" (attempts: {', '.join(attempts)})"
            raise SecurityError(f"Deserialization blocked by policy: {msg}") from last_exc
    obj = _resolve_trusted_loader(
        obj,
        auto_accept=auto_accept,
        backend=backend,
        trust_loader=trust_loader,
        envelope_path=getattr(envelope, "_path", None),
    )
    return obj


def _resolve_trusted_loader(
    obj: Any,
    *,
    auto_accept: bool = False,
    backend=None,
    trust_loader: Optional[bool] = None,
    envelope_path: str | Path | None = None,
) -> Any:
    """
    Resolve trusted-loader descriptors, prompting before execution.

    Also unwraps _beaver_figure dicts back to CapturedFigure for nice display.

    Args:
        obj: The object to resolve
        auto_accept: If True, automatically accept trusted loaders without prompting
        backend: Optional SyftBoxBackend for reading encrypted artifact files
        trust_loader: If True, run loader in trusted mode (full builtins).
                     If False, use RestrictedPython.
                     If None (default), try RestrictedPython first and prompt on failure.
    """
    try:
        from .twin import CapturedFigure, Twin  # local import to avoid cycles

        twin_cls = Twin
        captured_figure_cls = CapturedFigure
    except Exception:
        twin_cls = None  # type: ignore
        captured_figure_cls = None  # type: ignore

    if twin_cls is not None and isinstance(obj, twin_cls):
        obj.public = _resolve_trusted_loader(
            obj.public,
            auto_accept=auto_accept,
            backend=backend,
            trust_loader=trust_loader,
            envelope_path=envelope_path,
        )
        obj.private = _resolve_trusted_loader(
            obj.private,
            auto_accept=auto_accept,
            backend=backend,
            trust_loader=trust_loader,
            envelope_path=envelope_path,
        )
        # Also resolve captured figure lists (they contain _beaver_figure dicts)
        if hasattr(obj, "public_figures") and obj.public_figures:
            obj.public_figures = _resolve_trusted_loader(
                obj.public_figures,
                auto_accept=auto_accept,
                backend=backend,
                trust_loader=trust_loader,
                envelope_path=envelope_path,
            )
        if hasattr(obj, "private_figures") and obj.private_figures:
            obj.private_figures = _resolve_trusted_loader(
                obj.private_figures,
                auto_accept=auto_accept,
                backend=backend,
                trust_loader=trust_loader,
                envelope_path=envelope_path,
            )
        return obj

    # Handle inline parquet bytes (pyfory serializes pandas DataFrames as parquet)
    if twin_cls is not None and twin_cls._is_inline_parquet_bytes(obj):
        result = twin_cls._load_inline_parquet(obj)
        if result is not None:
            return result
        # Fall through if deserialization failed

    # Unwrap _beaver_figure dicts back to CapturedFigure for nice Jupyter display
    if isinstance(obj, dict) and obj.get("_beaver_figure") and captured_figure_cls is not None:
        return captured_figure_cls(obj)

    # Reconstruct pandas DataFrame from serialized dict
    if isinstance(obj, dict) and obj.get("_beaver_dataframe"):
        try:
            import pandas as pd

            return pd.DataFrame(obj["data"])
        except ImportError:
            # pandas not available, return the raw data
            return obj["data"]

    # Reconstruct pandas Series from serialized dict
    if isinstance(obj, dict) and obj.get("_beaver_series"):
        try:
            import pandas as pd

            return pd.Series(obj["data"], name=obj.get("name"))
        except ImportError:
            # pandas not available, return the raw data
            return obj["data"]

    if isinstance(obj, dict) and obj.get("_trusted_loader"):
        name = obj.get("name")
        data_path = obj.get("path")
        src = obj.get("deserializer_src", "")
        meta = obj.get("meta")  # Embedded metadata (e.g., pandas kind/name)

        backend_data_dir = None
        if backend is not None and getattr(backend, "data_dir", None) is not None:
            backend_data_dir = Path(backend.data_dir).resolve()

        # Resolve syft:// URLs to local datasites paths
        if data_path:
            from .syft_url import datasites_root_from_path, is_syft_url, syft_url_to_local_path

            if is_syft_url(data_path):
                datasites_root = None
                if backend_data_dir is not None:
                    datasites_root = backend_data_dir / "datasites"
                elif envelope_path is not None:
                    datasites_root = datasites_root_from_path(Path(envelope_path))
                if datasites_root is None:
                    raise SecurityError(
                        f"Cannot resolve syft:// path without datasites root: {data_path}"
                    )
                try:
                    data_path = str(
                        syft_url_to_local_path(data_path, datasites_root=datasites_root)
                    )
                except Exception as exc:
                    raise SecurityError(
                        f"Invalid syft:// path in trusted loader: {data_path}"
                    ) from exc

        # Translate path from sender's perspective to local view
        # If path contains /datasites/<identity>/, map to our local view
        # ALWAYS translate when backend is available - file is synced to our local view
        if data_path and backend and backend_data_dir is not None:
            stored = str(data_path)
            normalized = stored.replace("\\", "/")
            lower = normalized.lower()
            marker = "/datasites/"
            idx = lower.find(marker)
            if idx >= 0:
                relative = normalized[idx + len(marker) :].lstrip("/")
                parts = [p for p in relative.split("/") if p]
                if len(parts) >= 2 and parts[1] in {"shared", "app_data", "public"}:
                    data_path = str(Path(backend_data_dir) / "datasites" / relative)
            elif normalized.startswith("datasites/"):
                data_path = str(Path(backend_data_dir) / normalized)
            else:
                # Newer on-wire form may already be relative to datasites root:
                #   <email>/shared/... or <email>/app_data/... or <email>/public/...
                parts = [p for p in normalized.split("/") if p]
                if len(parts) >= 2 and parts[1] in {"shared", "app_data", "public"}:
                    data_path = str(Path(backend_data_dir) / "datasites" / normalized)

        # Resolve relative unix-style paths against the envelope's directory when available.
        if data_path and envelope_path is not None:
            import re

            stored = str(data_path)
            normalized = stored.replace("\\", "/")
            is_absolute = bool(
                normalized.startswith("/")
                or normalized.startswith("\\\\")
                or re.match(r"^[A-Za-z]:", normalized)
            )
            if not is_absolute:
                parts = [p for p in normalized.split("/") if p]
                if ".." in parts:
                    raise SecurityError(f"Path traversal detected for artifact: {data_path}")
                # If the envelope lives in a session `data/` folder, treat paths as relative
                # to the session directory (so artifacts use `data/<name>.bin`).
                envelope_dir = Path(envelope_path).resolve().parent
                session_base = envelope_dir.parent if envelope_dir.name == "data" else envelope_dir
                candidate = session_base.joinpath(*parts)
                # Back-compat: if older payloads stored filenames relative to `data/`, accept them.
                if not candidate.exists():
                    candidate_in_data = (session_base / "data").joinpath(*parts)
                    if candidate_in_data.exists():
                        candidate = candidate_in_data

                # If the envelope was read from the unencrypted shadow cache, prefer the
                # encrypted datasites path (read_with_shadow decrypts from encrypted path).
                if backend_data_dir is not None:
                    shadow_root = (backend_data_dir / "unencrypted").resolve()
                    datasites_root = (backend_data_dir / "datasites").resolve()
                    with contextlib.suppress(Exception):
                        rel_from_shadow = candidate.resolve().relative_to(shadow_root)
                        candidate2 = (datasites_root / rel_from_shadow).resolve()
                        if candidate2.exists():
                            candidate = candidate2

                data_path = str(candidate)

        # Validate artifact path to prevent traversal/symlink escapes
        if data_path:
            _validate_artifact_path(data_path, backend=backend)

        defined_functions = _loader_defined_functions(src)

        preview = "\n".join(src.strip().splitlines()[:5])
        auto_accept_env = os.getenv("BEAVER_AUTO_ACCEPT", "").lower() in {"1", "true", "yes"}
        non_interactive = False
        try:
            import sys

            non_interactive = not sys.stdin or not sys.stdin.isatty() or "ipykernel" in sys.modules
        except Exception:
            non_interactive = True

        effective_auto_accept = auto_accept or auto_accept_env or non_interactive
        if not effective_auto_accept:
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
        scope: dict[str, Any] = {"TrustedLoader": TrustedLoader, "Path": Path}
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

        # Inject metadata if available (eliminates need for .meta.json files)
        # Note: avoid underscore prefix as RestrictedPython blocks such names
        if meta is not None:
            scope["beaver_meta"] = meta

        # Static analysis to block dangerous imports/calls
        _analyze_loader_source(src, allowed_imports=_ALLOWED_LOADER_MODULES)

        trusted_loader_env = os.getenv("BEAVER_TRUSTED_LOADERS", "").lower() in {
            "1",
            "true",
            "yes",
        }

        def _run_trusted_loader():
            """Execute loader in trusted mode with guarded imports."""
            nonlocal scope

            def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
                top = name.split(".")[0]
                if top not in _ALLOWED_LOADER_MODULES:
                    raise SecurityError(f"Import not allowed in trusted loader: {name}")
                return __import__(name, globals, locals, fromlist, level)

            safe_builtins = {
                "len": len,
                "range": range,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "dict": dict,
                "list": list,
                "tuple": tuple,
                "set": set,
                "bytes": bytes,
                "object": object,
                "type": type,
                "isinstance": isinstance,
                "print": print,
                "Path": Path,
                "__import__": _guarded_import,
                "globals": lambda: scope,
                "locals": lambda: scope,
            }
            scope["__builtins__"] = safe_builtins
            exec(src, scope, scope)

        def _prompt_for_trust(error_msg: str) -> bool:
            """Prompt user to approve trusted mode execution."""
            import sys

            # Check if we're in non-interactive mode
            try:
                is_interactive = sys.stdin and sys.stdin.isatty() and "ipykernel" not in sys.modules
            except Exception:
                is_interactive = False

            if not is_interactive:
                return False

            print(f"\n⚠️  Loader '{name}' failed in restricted mode: {error_msg}")
            print(f"   Source preview:\n{preview}")
            print(f"   Data path: {data_path}")
            resp = input("   Allow trusted execution? [y/N]: ").strip().lower()
            return resp in ("y", "yes")

        # Determine execution mode
        use_trusted = trust_loader is True or trusted_loader_env

        if use_trusted:
            # Trusted mode requested explicitly
            try:
                _run_trusted_loader()
            except Exception as e:
                raise SecurityError(f"Trusted loader execution blocked: {e}") from e
        else:
            # Try RestrictedPython first, prompt on failure if trust_loader is None
            restricted_error = None
            try:
                scope = _exec_restricted_loader(src, scope)
            except Exception as e:
                restricted_error = e

            if restricted_error is not None:
                # RestrictedPython failed - check if we should prompt or fail
                if trust_loader is False:
                    # Explicitly disabled trusted mode
                    raise SecurityError(
                        f"Loader execution failed in restricted mode: {restricted_error}"
                    ) from restricted_error

                # trust_loader is None - prompt user
                if _prompt_for_trust(str(restricted_error)):
                    try:
                        _run_trusted_loader()
                    except Exception as e:
                        raise SecurityError(f"Trusted loader execution blocked: {e}") from e
                else:
                    raise SecurityError(
                        f"Loader execution failed in restricted mode (trusted mode not approved): "
                        f"{restricted_error}"
                    ) from restricted_error

        # Merge back safe symbols from scope (exclude modules, frames, builtins, and private names)
        safe_symbols = {}
        for key, val in scope.items():
            if key.startswith("_"):
                continue
            if key in {"TrustedLoader", "ad", "pd", "np", "anndata", "pandas", "numpy"}:
                continue
            if isinstance(val, (types.ModuleType, types.FrameType)):
                continue
            safe_symbols[key] = val

        # Attach merged symbols onto scope to allow callers to access derived functions/objects
        scope.update(safe_symbols)
        # Exclude TrustedLoader class, Path, and imported modules - we want the actual deserializer function
        excluded = {
            TrustedLoader,
            Path,
            scope.get("ad"),
            scope.get("pd"),
            scope.get("np"),
            scope.get("anndata"),
            scope.get("pandas"),
            scope.get("numpy"),
        }
        deser_fn = None
        for fn_name in defined_functions:
            candidate = scope.get(fn_name)
            if callable(candidate) and candidate not in excluded:
                deser_fn = candidate
                break
        if deser_fn is None:
            deser_fn = next((v for v in scope.values() if callable(v) and v not in excluded), None)
        if deser_fn is None:
            raise RuntimeError("No deserializer found in loader source")

        # If backend is available and uses crypto, read through backend (decrypts) and write to temp
        actual_path = data_path
        temp_file = None
        if backend and backend.uses_crypto:
            # Wait for file to appear AND stabilize (sync in progress) with timeout
            import time as _time

            spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            spin_idx = 0

            sync_timeout = 60.0  # seconds to wait for sync (increased for large files)
            sync_poll = 1.0  # poll interval
            stable_time = 2.0  # file size must be stable for this long
            sync_deadline = _time.monotonic() + sync_timeout
            waited = False
            last_size = -1
            stable_since = None
            last_print = 0.0
            spinner_interval = 0.2

            while True:
                now = _time.monotonic()
                if now >= sync_deadline:
                    raise RuntimeError(
                        f"Artifact file not synced at {data_path} after waiting {sync_timeout}s. "
                        f"File may still be syncing - try again later."
                    )

                file_path = Path(data_path)
                # Use cross-platform filename extraction for display (handles Windows paths on POSIX)
                display_filename = _cross_platform_filename(data_path)
                if not file_path.exists():
                    if not waited:
                        print(
                            f"⏳ Waiting for artifact file to sync: {display_filename} (at {file_path})"
                        )
                        waited = True
                    if now - last_print >= spinner_interval:
                        spin_idx = (spin_idx + 1) % len(spinner_chars)
                        print(f"{spinner_chars[spin_idx]} waiting for {display_filename}", end="\r")
                        last_print = now
                    _time.sleep(sync_poll)
                    continue

                # File exists - check if size has stabilized
                current_size = file_path.stat().st_size
                if current_size != last_size:
                    last_size = current_size
                    stable_since = now
                    if not waited:
                        print(
                            f"⏳ Waiting for artifact file to sync: {display_filename} "
                            f"(at {file_path}, {current_size:,} bytes)"
                        )
                        waited = True
                    if now - last_print >= spinner_interval:
                        spin_idx = (spin_idx + 1) % len(spinner_chars)
                        print(
                            f"{spinner_chars[spin_idx]} syncing {display_filename} ({current_size:,} bytes)",
                            end="\r",
                        )
                        last_print = now
                    _time.sleep(sync_poll)
                    continue

                # Size is same - check if stable long enough
                if stable_since and (now - stable_since) >= stable_time:
                    break  # File is stable, proceed

                if now - last_print >= spinner_interval:
                    spin_idx = (spin_idx + 1) % len(spinner_chars)
                    print(
                        f"{spinner_chars[spin_idx]} syncing {display_filename} ({current_size:,} bytes)",
                        end="\r",
                    )
                    last_print = now
                _time.sleep(sync_poll)

            if waited:
                print(
                    f"\n✓ Artifact file synced: {_cross_platform_filename(data_path)} ({last_size:,} bytes)"
                )
            else:
                print()

            # File exists, decrypt it
            try:
                data = backend.read_bytes(data_path)
                temp_file = Path(tempfile.gettempdir()) / f"_beaver_read_{uuid4().hex}.bin"
                temp_file.write_bytes(data)
                actual_path = str(temp_file)
            except Exception as e:
                # Log the error but still try to decrypt - the file is encrypted
                import beaver

                if beaver.debug:
                    print(f"[DEBUG] Failed to decrypt {data_path}: {e}")
                # Don't fall back to direct read for encrypted files
                raise RuntimeError(f"Failed to decrypt artifact file {data_path}: {e}") from e

        try:
            return deser_fn(actual_path)
        finally:
            # Clean up temp file
            if temp_file and temp_file.exists():
                temp_file.unlink()

    if isinstance(obj, list):
        return [
            _resolve_trusted_loader(
                x,
                auto_accept=auto_accept,
                backend=backend,
                trust_loader=trust_loader,
                envelope_path=envelope_path,
            )
            for x in obj
        ]
    if isinstance(obj, tuple):
        return tuple(
            _resolve_trusted_loader(
                x,
                auto_accept=auto_accept,
                backend=backend,
                trust_loader=trust_loader,
                envelope_path=envelope_path,
            )
            for x in obj
        )
    if isinstance(obj, dict):
        return {
            k: _resolve_trusted_loader(
                v,
                auto_accept=auto_accept,
                backend=backend,
                trust_loader=trust_loader,
                envelope_path=envelope_path,
            )
            for k, v in obj.items()
        }
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
    # Merge Twins by twin_id when an existing instance is already present.
    # Only fill in sides (public/private) that are currently missing to avoid clobbering local data.
    try:
        from .twin import _TWIN_REGISTRY as twin_registry  # noqa: N811
        from .twin import Twin as twin_cls  # noqa: N813
    except Exception:
        twin_cls = None
        twin_registry = {}

    if twin_cls and isinstance(obj, twin_cls):
        twin_id = getattr(obj, "twin_id", None)
        existing = None

        # Check globals for an existing Twin with the same id
        if twin_id:
            for val in globals_ns.values():
                if isinstance(val, twin_cls) and getattr(val, "twin_id", None) == twin_id:
                    existing = val
                    break

        # Fallback to registry
        if existing is None and twin_id:
            for (tid, _owner), tw in twin_registry.items():
                if tid == twin_id:
                    existing = tw
                    break

        def _merge_twin(dst: twin_cls, src: twin_cls):
            # Lazy import to avoid cycles
            try:
                from .computation import ComputationRequest as computation_request_cls  # noqa: N813
            except Exception:
                computation_request_cls = None

            # If dst holds a request placeholder and src carries the matching result twin_id,
            # treat them as the same Twin and update the ID to the resolved one.
            dst_private = getattr(dst, "private", None)
            placeholder_id = None
            if computation_request_cls is not None and isinstance(
                dst_private, computation_request_cls
            ):
                placeholder_id = getattr(dst_private, "result_id", None)
                if placeholder_id and getattr(src, "twin_id", None) == placeholder_id:
                    with contextlib.suppress(Exception):
                        del twin_registry[(dst.twin_id, dst.owner)]
                    dst.twin_id = src.twin_id
                    twin_registry[(dst.twin_id, dst.owner)] = dst

            # Fill in missing sides only
            if getattr(dst, "public", None) is None and getattr(src, "public", None) is not None:
                dst.public = src.public
                if hasattr(src, "public_stdout"):
                    dst.public_stdout = getattr(src, "public_stdout", None)
                if hasattr(src, "public_stderr"):
                    dst.public_stderr = getattr(src, "public_stderr", None)
                if hasattr(src, "public_figures"):
                    dst.public_figures = getattr(src, "public_figures", None)

            # Treat a ComputationRequest placeholder as "missing private"
            private_missing = dst_private is None or (
                computation_request_cls is not None
                and isinstance(dst_private, computation_request_cls)
            )

            # Check if this is an idempotent reload (same result loaded twice)
            # This happens when auto-load updates the Twin, then user manually loads from inbox
            src_private = getattr(src, "private", None)
            is_idempotent_reload = False
            if dst_private is not None and src_private is not None:
                # Both have values - check if they're compatible (same type)
                from .twin import CapturedFigure

                # If both are CapturedFigure or same basic type, allow idempotent reload
                if (
                    isinstance(dst_private, CapturedFigure)
                    and isinstance(src_private, CapturedFigure)
                ) or (type(dst_private).__name__ == type(src_private).__name__):
                    is_idempotent_reload = True

            def _log_merge(action: str, reason: str):
                print(
                    f"🔄 Twin merge {action} private "
                    f"(dst_id={dst.twin_id[:12]}..., src_id={getattr(src, 'twin_id', '')[:12]}..., "
                    f"dst_private={type(dst_private).__name__}, src_private={type(src_private).__name__}, "
                    f"placeholder={private_missing}, reason={reason})"
                )

            if src_private is not None:
                if (
                    private_missing
                    or getattr(src, "force_private_replace", False)
                    or is_idempotent_reload
                ):
                    if is_idempotent_reload and not private_missing:
                        _log_merge("updating", "idempotent_reload")
                    else:
                        _log_merge("updating", "missing_or_forced")
                    dst.private = src.private
                    if hasattr(src, "private_stdout"):
                        dst.private_stdout = getattr(src, "private_stdout", None)
                    if hasattr(src, "private_stderr"):
                        dst.private_stderr = getattr(src, "private_stderr", None)
                    if hasattr(src, "private_figures"):
                        dst.private_figures = getattr(src, "private_figures", None)
                else:
                    _log_merge("skipped", "dst_private_already_set")
            # Keep registry in sync
            twin_registry[(dst.twin_id, dst.owner)] = dst

        if existing:
            _merge_twin(existing, obj)
            obj = existing

            # Sync all Twin instances with same twin_id to point to canonical merged object
            # This returns the canonical Twin (which may be different from 'existing')
            from .twin import _sync_twin_instances

            canonical = _sync_twin_instances(twin_id, existing.owner, existing)
            if canonical is not None:
                obj = canonical  # Use the canonical Twin for injection
                # Update registry to point to canonical
                twin_registry[(twin_id, canonical.owner)] = canonical
        else:
            if twin_id:
                print(
                    f"⚠️ Twin merge: no existing Twin with id {twin_id[:12]}... "
                    f"found in globals/registry for owner '{obj.owner}'. "
                    "Creating new instance; public/private may stay split."
                )

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
) -> tuple[Optional[BeaverEnvelope], Optional[Any]]:
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


def list_inbox(inbox: Path | str, *, backend=None) -> list[BeaverEnvelope]:
    """List and load all .beaver envelopes in an inbox.

    Args:
        inbox: Path to inbox directory
        backend: Optional SyftBoxBackend for decrypting encrypted files

    Returns:
        List of BeaverEnvelope objects
    """
    inbox_path = Path(inbox)
    envelopes = []

    # Collect all .beaver files from root and data/ subdirectory
    beaver_files = list(inbox_path.glob("*.beaver"))
    data_dir = inbox_path / "data"
    if data_dir.exists():
        beaver_files.extend(data_dir.glob("*.beaver"))

    for p in sorted(beaver_files):
        try:
            if backend is not None and backend.uses_crypto:
                # Use backend to decrypt encrypted files
                env = backend.read_envelope(p)
            else:
                env = read_envelope(p)
            envelopes.append(env)
        except Exception:
            # Skip files that can't be read (may be partially synced or corrupt)
            pass
    return envelopes


class InboxView:
    """Wrapper to pretty-print an inbox as a table in notebooks."""

    def __init__(self, inbox: Path | str, envelopes: list[BeaverEnvelope], session=None) -> None:
        self.inbox = Path(inbox)
        self.envelopes = envelopes
        self.session = session  # Session reference for session-scoped inboxes

        # Attach session to each envelope so loaded objects get session context
        if session is not None:
            for env in self.envelopes:
                env._session = session

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
        "Tuple": tuple,
        "List": list,
        "Dict": dict,
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
) -> tuple[Optional[BeaverEnvelope], Optional[Any], Optional[Any]]:
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
                # Check for missing imports before execution
                from .computation import _check_missing_imports, _prompt_install_function_deps

                missing_imports = _check_missing_imports(func)
                if missing_imports and not _prompt_install_function_deps(
                    missing_imports, func.__name__
                ):
                    # User declined or installation failed - raise error
                    import shutil

                    use_uv = shutil.which("uv") is not None
                    pip_cmd = "uv pip install" if use_uv else "pip install"
                    deps_str = " ".join(missing_imports)
                    raise ImportError(
                        f"\n\n❌ Cannot execute '{func.__name__}' - missing dependencies\n\n"
                        f"   Required packages not installed: {', '.join(missing_imports)}\n\n"
                        f"   To fix, run:\n"
                        f"   {pip_cmd} {deps_str}\n"
                    )

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
                captured_fig_ids = set()

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

                    # Deduplicate captured figures by content
                    import hashlib

                    seen_signatures = set()
                    deduped_figures = []

                    def figure_signature(png_bytes: bytes):
                        """Stable signature of figure pixels to catch duplicates with different metadata."""
                        if not png_bytes:
                            return None
                        try:
                            from PIL import Image

                            with Image.open(io.BytesIO(png_bytes)) as img:
                                raw_bytes = img.tobytes()
                                digest = hashlib.sha256()
                                digest.update(str(img.size).encode())
                                digest.update(img.mode.encode())
                                digest.update(raw_bytes)
                                return digest.digest()
                        except Exception:
                            return hashlib.sha256(png_bytes).digest()

                    for fig in captured_figures:
                        png_bytes = fig.png_bytes if hasattr(fig, "png_bytes") else None
                        signature = figure_signature(png_bytes) if png_bytes else None
                        if signature and signature in seen_signatures:
                            continue
                        if signature:
                            seen_signatures.add(signature)
                        deduped_figures.append(fig)

                    captured_figures = deduped_figures

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
                # Convert Twins and TwinFuncs to references (pointer dicts)
                from .twin_func import TwinFunc
                from .twin_func import func_to_ref as _func_to_ref

                def twin_to_ref(val):
                    """Convert Twin/TwinFunc to pointer reference for serialization."""
                    if isinstance(val, Twin):
                        return {
                            "_beaver_twin_ref": True,
                            "twin_id": val.twin_id,
                            "owner": val.owner,
                            "name": val.name,
                            "syft_url": getattr(val, "syft_url", None),
                            "dataset_asset": getattr(val, "dataset_asset", None),
                        }
                    # Handle TwinFunc references
                    if isinstance(val, TwinFunc):
                        return val.to_ref()
                    # Handle registered callables with beaver metadata
                    if callable(val) and hasattr(val, "_beaver_func_id"):
                        return _func_to_ref(val)
                    return val

                ref_args = tuple(twin_to_ref(arg) for arg in args)
                ref_kwargs = {k: twin_to_ref(v) for k, v in kwargs.items()}

                comp_id = uuid4().hex
                # Detect required package versions for the function
                from .computation import _detect_function_imports_with_versions

                required_versions = _detect_function_imports_with_versions(func)

                comp_request = ComputationRequest(
                    comp_id=comp_id,
                    result_id=uuid4().hex,
                    func=func,
                    args=ref_args,  # Use references, not full objects
                    kwargs=ref_kwargs,
                    sender=context.user if context else "unknown",
                    result_name=name or f"{func.__name__}_result",
                    required_versions=required_versions,
                )

                # Handle None results - use sentinel dict to allow proper display
                # Functions that return None (like plotting functions) are valid
                public_value = public_result
                if public_value is None:
                    public_value = {"_none_result": True, "has_figures": len(captured_figures) > 0}

                # Find session from input Twins (if any)
                input_session = None
                for arg in args:
                    if (
                        isinstance(arg, Twin)
                        and hasattr(arg, "_session")
                        and arg._session is not None
                    ):
                        input_session = arg._session
                        break
                for val in kwargs.values():
                    if input_session:
                        break
                    if (
                        isinstance(val, Twin)
                        and hasattr(val, "_session")
                        and val._session is not None
                    ):
                        input_session = val._session
                        break

                # Attach session to ComputationRequest so it's available when executed
                if input_session is not None:
                    comp_request._session = input_session

                # Return Twin with public result and computation request
                result_twin = Twin(
                    public=public_value,
                    private=comp_request,  # Computation request, not result yet
                    owner=destination_user or "unknown",
                    name=name or f"{func.__name__}_result",
                    twin_id=comp_request.result_id,
                )

                # Inherit session from input Twins so request_private() uses session folder
                if input_session is not None:
                    result_twin._session = input_session

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
) -> tuple[Optional[BeaverEnvelope], Optional[Any]]:
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

            if self.context._backend:
                # SyftBox mode: peer's data is in datasites/<peer>/shared/
                registry_path = (
                    self.context._backend.data_dir
                    / "datasites"
                    / self.username
                    / "shared"
                    / "public"
                    / "remote_vars.json"
                )
                data_dir = (
                    self.context._backend.data_dir
                    / "datasites"
                    / self.username
                    / "shared"
                    / "biovault"
                )
            else:
                # Legacy mode
                registry_path = self.context._public_dir / self.username / "remote_vars.json"
                data_dir = self.context._public_dir / self.username / "data"

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
        poll_interval: float = 0.5,
        _backend=None,  # SyftBoxBackend instance for encrypted mode
    ) -> None:
        # If caller passed the default "unknown", try env so we avoid creating paths under "unknown"
        if user == "unknown":
            import os

            user = os.environ.get("SYFTBOX_EMAIL", user)

        self.inbox_path = Path(inbox)
        self.outbox = Path(outbox) if outbox is not None else self.inbox_path
        self.user = user
        self.biovault = biovault
        self.strict = strict
        self.policy = policy
        self._active_session = None  # Track the most recently created/accepted session

        # SyftBox backend (None for legacy filesystem mode)
        self._backend = _backend

        # Settings
        from .settings import BeaverSettings

        self._settings = BeaverSettings()

        # Remote vars setup
        if self._backend:
            # SyftBox mode: use backend's data_dir structure
            self._base_dir = self._backend.data_dir
            self._public_dir = self._base_dir / "datasites" / self.user / "shared" / "public"
            self._registry_path = self._public_dir / "remote_vars.json"
        else:
            # Legacy mode
            self._base_dir = self.inbox_path.parent.parent  # shared/
            self._public_dir = self._base_dir / "shared" / "public"
            self._registry_path = self._public_dir / self.user / "remote_vars.json"
        self._remote_vars_registry = None
        self._datasets = None
        self._mappings = None

        # Staging area for remote computations
        self._staging_area = None

        # Auto-load replies setup
        self._auto_load_enabled = False
        self._poll_interval = poll_interval
        self._auto_load_thread = None
        self._stop_auto_load = False
        self._processed_replies = set()  # Track processed envelope IDs

        if auto_load_replies:
            self.start_auto_load()

        # Set as current context for SyftBox-aware file operations
        _set_current_context(self)

    @property
    def settings(self):
        """
        Access configuration settings.

        Returns:
            BeaverSettings object for managing configuration
        """
        return self._settings

    @property
    def syftbox_enabled(self) -> bool:
        """Check if SyftBox encrypted mode is enabled."""
        return self._backend is not None

    @property
    def backend(self):
        """
        Access the SyftBox backend (if enabled).

        Returns:
            SyftBoxBackend instance or None
        """
        return self._backend

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

    @property
    def mappings(self):
        """
        Access private-url -> local-path mappings (BIOVAULT_HOME/mapping.yaml).
        """
        if self._mappings is None:
            from .mappings import MappingStore

            self._mappings = MappingStore.from_env(allow_missing=True)
        return self._mappings

    @property
    def datasets(self):
        """
        Access datasets published to SyftBox datasites.

        Returns:
            DatasetRegistry for browsing datasets and materializing Twins.
        """
        if self._datasets is None:
            from .datasets import DatasetRegistry

            self._datasets = DatasetRegistry(
                base_dir=self._base_dir,
                backend=self._backend,
                mapping_store=self.mappings,
            )
        return self._datasets

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

    # -------------------------------------------------------------------------
    # Session Management
    # -------------------------------------------------------------------------

    def session_requests(self):
        """
        List pending session requests from other users.

        Returns:
            SessionRequestsView with pending requests that can be accepted/rejected

        Example:
            requests = bv.session_requests()
            requests[0].accept()  # Accept first request
        """
        from .session import SessionRequestsView

        if self._backend is None:
            print("⚠️  Session management requires SyftBox mode")
            return SessionRequestsView([])

        requests = self._backend.list_session_requests()

        # Bind context to each request so .accept() works
        for req in requests:
            req._context = self

        return SessionRequestsView(requests)

    def request_session(
        self,
        peer_email: str,
        message: Optional[str] = None,
    ):
        """
        Request a new session with a data owner.

        Args:
            peer_email: Email of the data owner
            message: Optional message to include with request

        Returns:
            Session object (in pending state until accepted)

        Example:
            session = bv.request_session("client1@sandbox.local")
            session.wait_for_acceptance()
        """
        from .session import Session, _generate_session_id

        if self._backend is None:
            raise RuntimeError("Session management requires SyftBox mode")

        # Generate session ID
        session_id = _generate_session_id()

        # Send request via RPC
        self._backend.send_session_request(
            session_id=session_id,
            target_email=peer_email,
            message=message,
        )

        # Create local session object
        session = Session(
            session_id=session_id,
            peer=peer_email,
            owner=self.user,
            role="requester",
            status="pending",
        )
        session._context = self
        self._active_session = session

        print(f"📤 Session request sent to {peer_email}")
        print(f"   Session ID: {session_id}")
        print("   Use session.wait_for_acceptance() to wait for approval")

        return session

    def accept_session(self, request_or_id) -> Session:
        """
        Accept a session request.

        Args:
            request_or_id: SessionRequest object, session ID string, or requester email

        Returns:
            Session object for the accepted session

        Example:
            # Accept by request object
            bv.session_requests()[0].accept()

            # Accept by session ID
            bv.accept_session("abc123")

            # Accept by requester email
            bv.accept_session("client2@sandbox.local")
        """
        from .session import Session, SessionRequest

        if self._backend is None:
            raise RuntimeError("Session management requires SyftBox mode")

        # Resolve the request
        if isinstance(request_or_id, SessionRequest):
            request = request_or_id
        elif isinstance(request_or_id, str):
            # Find by ID or email
            requests = self._backend.list_session_requests()
            request = None
            for req in requests:
                if req.session_id == request_or_id or req.session_id.startswith(request_or_id):
                    request = req
                    break
                if req.requester == request_or_id:
                    request = req
                    break

            if request is None:
                raise KeyError(f"No pending session request matching: {request_or_id}")
        else:
            raise TypeError(f"Expected SessionRequest or str, got {type(request_or_id)}")

        # Send acceptance response
        self._backend.send_session_response(
            session_id=request.session_id,
            requester_email=request.requester,
            accepted=True,
        )

        # Create session folder
        self._backend.create_session_folder(
            session_id=request.session_id,
            peer_email=request.requester,
        )

        # Create local session object
        session = Session(
            session_id=request.session_id,
            peer=request.requester,
            owner=self.user,
            role="accepter",
            status="active",
        )
        session._context = self
        session._setup_paths()
        self._active_session = session

        print(f"✅ Session accepted: {request.session_id}")
        print(f"   Peer: {request.requester}")
        print(f"   Session folder: {session.local_folder}")

        return session

    def _reject_session(self, request, reason: Optional[str] = None) -> None:
        """Reject a session request (internal)."""
        if self._backend is None:
            raise RuntimeError("Session management requires SyftBox mode")

        self._backend.send_session_response(
            session_id=request.session_id,
            requester_email=request.requester,
            accepted=False,
            reason=reason,
        )

        print(f"❌ Session rejected: {request.session_id}")

    def active_session(self) -> Optional[Session]:
        """
        Get the currently active session, if any.

        Checks for an active session in this order:
        1. Previously created/accepted session in this context
        2. BEAVER_SESSION_ID environment variable
        3. session.json file in current working directory

        Returns:
            Session object if an active session exists, None otherwise

        Example:
            bv = beaver.connect()
            session = bv.active_session()
            if session:
                session.send(my_data)
        """
        import os
        from pathlib import Path

        from .session import Session

        # 1. Check if we have an active session from this context
        if self._active_session and self._active_session.is_active:
            return self._active_session

        # 2. Check BEAVER_SESSION_ID environment variable
        session_id = os.environ.get("BEAVER_SESSION_ID")

        # 3. Check session.json in cwd
        session_json_path = Path(os.getcwd()) / "session.json"
        session_config = None

        if session_json_path.exists():
            try:
                import json

                with open(session_json_path) as f:
                    session_config = json.load(f)
                    if not session_id:
                        session_id = session_config.get("session_id")
            except Exception as e:
                print(f"⚠️  Failed to read session.json: {e}")

        if not session_id:
            return None

        # Try to load session from config or reconstruct from session_id
        peer = None
        role = None
        status = "active"

        if session_config:
            peer = session_config.get("peer")
            role = session_config.get("role", "accepter")
            status = session_config.get("status", "active")

        # If we don't have peer info, try to find it from RPC files
        if not peer and self._backend:
            # Check our RPC folder for matching request/response
            rpc_path = self._backend.get_rpc_session_path()
            if rpc_path.exists():
                # Look for request file
                request_file = rpc_path / f"{session_id}.request"
                if request_file.exists():
                    try:
                        if self._backend.uses_crypto:
                            data = bytes(self._backend.storage.read_with_shadow(str(request_file)))
                            req_data = json.loads(data.decode("utf-8"))
                        else:
                            req_data = json.loads(request_file.read_text())

                        peer = req_data.get("requester")
                        role = "accepter"
                    except Exception:
                        pass

                # Look for response file (means we sent a request)
                response_file = rpc_path / f"{session_id}.response"
                if response_file.exists() and not peer:
                    try:
                        if self._backend.uses_crypto:
                            data = bytes(self._backend.storage.read_with_shadow(str(response_file)))
                            resp_data = json.loads(data.decode("utf-8"))
                        else:
                            resp_data = json.loads(response_file.read_text())

                        peer = resp_data.get("responder")
                        role = "requester"
                        status = resp_data.get("status", "active")
                    except Exception:
                        pass

        if not peer:
            # Solo session - use self as peer for local testing
            peer = self.user
            if not peer:
                print(f"⚠️  Session {session_id} found but peer info missing")
                return None

        if status != "active" and status != "accepted":
            print(f"⚠️  Session {session_id} is not active (status: {status})")
            return None

        # Create and setup the session
        session = Session(
            session_id=session_id,
            peer=peer,
            owner=self.user,
            role=role or "accepter",
            status="active",
        )
        session._context = self
        session._setup_paths()
        self._active_session = session

        print(f"🟢 Active session loaded: {session_id}")
        print(f"   Peer: {peer}")

        return session

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
        envelopes = list_inbox(self.inbox_path, backend=self._backend)

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

        # Use encrypted backend if available, otherwise plain write
        if self._backend and to_user:
            # Encrypt for the recipient
            path = self._backend.write_envelope(
                envelope=env,
                recipients=[to_user],
            )
        else:
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
        include_session: bool = True,
        check_existing: bool = True,
    ):
        """
        Wait for a new message to arrive in the inbox.

        Args:
            timeout: Max seconds to wait (default 60)
            poll_interval: Seconds between checks (default 1)
            filter_sender: Only wait for messages from this sender
            filter_name: Only wait for messages with this name
            auto_load: If True, automatically load and return the object
            include_session: Also watch the active session folder (if any)
            check_existing: If True, first check existing messages (default True)

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
        candidate_inboxes = self._candidate_inboxes(include_session=include_session)

        def _matches_filter(env):
            if filter_sender and env.sender != filter_sender:
                return False
            return not (filter_name and env.name != filter_name)

        # First, check existing messages (in case message arrived before we started waiting)
        if check_existing:
            for inbox_path in candidate_inboxes:
                for env in list_inbox(inbox_path, backend=self._backend):
                    if _matches_filter(env):
                        print(f"📬 Found existing message: {env.name or env.envelope_id[:12]}")
                        print(f"   From: {env.sender}")
                        if auto_load:
                            obj = unpack(
                                env,
                                strict=self.strict,
                                policy=self.policy,
                                backend=self._backend,
                            )
                            return env, obj
                        return env

        # No existing match - wait for new ones
        seen_ids = set()
        for inbox_path in candidate_inboxes:
            seen_ids.update(
                env.envelope_id for env in list_inbox(inbox_path, backend=self._backend)
            )

        if filter_name:
            print(f"⏳ Waiting for message with name '{filter_name}'...")
        elif filter_sender:
            print(f"⏳ Waiting for message from '{filter_sender}'...")
        else:
            print("⏳ Waiting for any message...")

            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                for inbox_path in candidate_inboxes:
                    for env in list_inbox(inbox_path, backend=self._backend):
                        if env.envelope_id in seen_ids:
                            continue  # Already seen
                        seen_ids.add(env.envelope_id)

                        if _matches_filter(env):
                            print(f"📬 New message: {env.name or env.envelope_id[:12]}")
                            print(f"   From: {env.sender}")

                            if auto_load:
                                obj = unpack(
                                    env,
                                    strict=self.strict,
                                    policy=self.policy,
                                    backend=self._backend,
                                )
                                return env, obj
                            return env

            time.sleep(poll_interval)

        print(f"⏰ Timeout after {timeout}s waiting for message")
        return None, None if auto_load else None

    def wait_for_request(
        self,
        twin=None,
        *,
        timeout: float = 120.0,
        poll_interval: float = 0.5,
        include_session: bool = True,
    ):
        """
        Wait for a computation request, optionally for a specific Twin.

        Args:
            twin: Twin object or name to wait for requests on (None = any request)
            timeout: Max seconds to wait (default 120)
            poll_interval: Seconds between checks (default 2)
            include_session: Also watch the active session folder (if any)

        Returns:
            The loaded ComputationRequest object, ready to run

        Example:
            # Wait for any request
            request = bv.wait_for_request()
            result = request.run_both()

            # Wait for request on specific Twin
            number = Twin(private=1337, public=100, name="number")
            session.remote_vars["number"] = number

            request = bv.wait_for_request(number)
            result = request.run_both()
            result.approve()
        """
        # Determine what we're looking for
        target_name = None
        if twin is not None:
            if isinstance(twin, str):
                target_name = twin
            elif hasattr(twin, "name") and twin.name:
                target_name = twin.name

        candidate_inboxes = self._candidate_inboxes(include_session=include_session)

        # Track processed requests to avoid returning same request twice
        if not hasattr(self, "_processed_request_ids"):
            self._processed_request_ids = set()

        def _is_processed(env):
            """Check if this request was already processed."""
            return env.envelope_id in self._processed_request_ids

        def _mark_processed(env):
            """Mark request as processed so we don't return it again."""
            self._processed_request_ids.add(env.envelope_id)

        def _matches_target_fast(env):
            """Quick check if envelope might match our target (metadata only)."""
            if not env.name or not env.name.startswith("request_"):
                return False
            if _is_processed(env):
                return False
            if not target_name:
                return True
            # Check env.inputs list (contains input variable names)
            inputs_match = target_name in (env.inputs or [])
            # Also check signature in manifest as fallback
            signature = env.manifest.get("signature", "") if env.manifest else ""
            signature_match = (
                f"({target_name})" in signature
                or f", {target_name})" in signature
                or f"({target_name}," in signature
            )
            return inputs_match or signature_match

        def _matches_target_deep(obj, env):
            """Deep check if unpacked ComputationRequest matches target by checking args."""
            if _is_processed(env):
                return False
            if not target_name:
                return True
            if hasattr(obj, "args"):
                for arg in obj.args:
                    if isinstance(arg, dict) and arg.get("name") == target_name:
                        return True
            return False

        # First, check for existing matching requests (return immediately if found)
        # Try fast match first, then deep match for backwards compatibility
        # Use TRUSTED_POLICY since ComputationRequest contains functions (for review, not execution)
        for inbox_path in candidate_inboxes:
            for env in list_inbox(inbox_path, backend=self._backend):
                if _matches_target_fast(env):
                    print(f"📬 Found existing request: {env.name}")
                    print(f"   From: {env.sender}")
                    obj = unpack(
                        env,
                        strict=self.strict,
                        policy=TRUSTED_POLICY,
                        backend=self._backend,
                    )
                    _mark_processed(env)
                    return obj

        # If target specified but no fast match, try deep matching (unpack and check args)
        if target_name:
            for inbox_path in candidate_inboxes:
                for env in list_inbox(inbox_path, backend=self._backend):
                    if env.name and env.name.startswith("request_") and not _is_processed(env):
                        obj = unpack(
                            env,
                            strict=self.strict,
                            policy=TRUSTED_POLICY,
                            backend=self._backend,
                        )
                        if _matches_target_deep(obj, env):
                            print(f"📬 Found existing request: {env.name}")
                            print(f"   From: {env.sender}")
                            _mark_processed(env)
                            return obj

        # No existing match - now wait for new ones
        # Track already-seen to avoid re-processing
        seen_ids = set()
        for inbox_path in candidate_inboxes:
            for env in list_inbox(inbox_path, backend=self._backend):
                seen_ids.add(env.envelope_id)

        if target_name:
            print(f"⏳ Waiting for request on '{target_name}'...")
        else:
            print("⏳ Waiting for any computation request...")

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for inbox_path in candidate_inboxes:
                for env in list_inbox(inbox_path, backend=self._backend):
                    # Skip already-seen in this wait loop
                    if env.envelope_id in seen_ids:
                        continue
                    seen_ids.add(env.envelope_id)

                    # Try fast match first (includes _is_processed check)
                    if _matches_target_fast(env):
                        print(f"📬 Request received: {env.name}")
                        print(f"   From: {env.sender}")
                        obj = unpack(
                            env,
                            strict=self.strict,
                            policy=TRUSTED_POLICY,
                            backend=self._backend,
                        )
                        _mark_processed(env)
                        return obj

                    # For new requests, also try deep match if target specified
                    if (
                        target_name
                        and env.name
                        and env.name.startswith("request_")
                        and not _is_processed(env)
                    ):
                        obj = unpack(
                            env,
                            strict=self.strict,
                            policy=TRUSTED_POLICY,
                            backend=self._backend,
                        )
                        if _matches_target_deep(obj, env):
                            print(f"📬 Request received: {env.name}")
                            print(f"   From: {env.sender}")
                            _mark_processed(env)
                            return obj

            time.sleep(poll_interval)

        print(f"⏰ Timeout after {timeout}s waiting for request")
        return None

    def wait_for_response(
        self,
        twin,
        *,
        timeout: float = 120.0,
        poll_interval: float = 0.5,
        include_session: bool = True,
    ):
        """
        Wait for a response to a computation request on a Twin.

        Call this after twin.request_private() to wait for the result.

        Args:
            twin: Twin object that has a pending request
            timeout: Max seconds to wait (default 120)
            poll_interval: Seconds between checks (default 2)
            include_session: Also watch the active session folder (if any)

        Returns:
            The Twin with updated private value, or None on timeout

        Example:
            # DS sends request
            res = analyze(peer_twin)
            res.request_private()

            # Wait for DO to approve
            result = bv.wait_for_response(res)
            print(result.private)  # The approved result!
        """
        # Get the pending computation ID
        comp_id = getattr(twin, "_pending_comp_id", None)
        twin_name = getattr(twin, "name", None) or "result"

        if comp_id is None:
            print("⚠️  No pending request on this Twin. Call .request_private() first.")
            return None

        candidate_inboxes = self._candidate_inboxes(include_session=include_session)

        def _handle_response(env):
            """Process a matching response envelope."""
            print(f"📬 Response received for '{twin_name}'")
            print(f"   From: {env.sender}")

            # Load the response
            obj = unpack(env, strict=self.strict, policy=self.policy, backend=self._backend)

            # Update the twin's private value and captured outputs
            if hasattr(obj, "private"):
                twin.private = obj.private
                # Also copy captured outputs from the response
                for attr in ("private_stdout", "private_stderr", "private_figures"):
                    if hasattr(obj, attr):
                        setattr(twin, attr, getattr(obj, attr))
            else:
                twin.private = obj

            # Clear pending state
            twin._pending_comp_id = None

            print(f"✅ '{twin_name}' updated with result")
            return twin

        # First, check for existing matching response
        for inbox_path in candidate_inboxes:
            for env in list_inbox(inbox_path, backend=self._backend):
                if env.reply_to == comp_id:
                    return _handle_response(env)

        # No existing match - wait for new ones
        seen_ids = set()
        for inbox_path in candidate_inboxes:
            for env in list_inbox(inbox_path, backend=self._backend):
                seen_ids.add(env.envelope_id)

        print(f"⏳ Waiting for response to '{twin_name}'...")
        print(f"   Looking for reply_to: {comp_id[:12]}...")
        print(f"   Watching: {[str(p) for p in candidate_inboxes]}")

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for inbox_path in candidate_inboxes:
                for env in list_inbox(inbox_path, backend=self._backend):
                    if env.envelope_id in seen_ids:
                        continue
                    seen_ids.add(env.envelope_id)

                    if env.reply_to == comp_id:
                        return _handle_response(env)

            time.sleep(poll_interval)

        print(f"⏰ Timeout after {timeout}s waiting for response")
        return None

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
        include_session = kwargs.pop("include_session", False)
        inbox_path = kwargs.get("inbox", self.inbox_path)
        envelopes = []
        for candidate in self._candidate_inboxes(
            include_session=include_session, base_path=inbox_path
        ):
            envelopes.extend(list_inbox(candidate, backend=self._backend))
        return envelopes

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
        include_session: bool = True,
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
            include_session: Include envelopes from the active session folder (if any)

        Returns:
            InboxView with sorted envelopes
        """
        inbox_path = kwargs.get("inbox", self.inbox_path)
        envelopes = []
        for candidate in self._candidate_inboxes(
            include_session=include_session, base_path=inbox_path
        ):
            envelopes.extend(list_inbox(candidate, backend=self._backend))

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _candidate_inboxes(self, *, include_session: bool, base_path: Optional[Path] = None):
        """
        Build a list of inbox paths to watch. By default, this is the base inbox
        plus the active session's peer folder (where requests/results land).
        Also includes peer's root biovault folder for fallback case when results
        are sent to inbox instead of session folder.
        """
        paths = []
        base = base_path or self.inbox_path
        paths.append(base)

        if include_session and self._active_session and self._active_session.is_active:
            if self._active_session.peer_folder is None:
                self._active_session._setup_paths()
            if self._active_session.peer_folder:
                paths.append(self._active_session.peer_folder)
                # Also check peer's root biovault folder (for results sent to inbox instead of session)
                peer_root = self._active_session.peer_folder.parent.parent
                if peer_root.exists() and peer_root != self._active_session.peer_folder:
                    paths.append(peer_root)

        # Deduplicate while preserving order
        unique = []
        seen = set()
        for p in paths:
            if p not in seen:
                unique.append(p)
                seen.add(p)
        return unique

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
    folder: Optional[Path | str] = None,
    *,
    user: str = "unknown",
    biovault: Optional[str] = None,
    inbox: Optional[Path | str] = None,
    outbox: Optional[Path | str] = None,
    strict: bool = False,
    policy=None,
    auto_load_replies: bool = True,
    poll_interval: float = 0.5,
    # SyftBox parameters
    syftbox: bool = True,  # Default to SyftBox mode
    data_dir: Optional[Path | str] = None,
    vault_path: Optional[Path | str] = None,
    disable_crypto: bool = False,
) -> BeaverContext:
    """
    Create a BeaverContext with shared defaults.

    Args:
        folder: Base directory for shared files (only used if syftbox=False)
        user: Username/email for this context (e.g., "client1@sandbox.local")
        biovault: Optional biovault identifier
        inbox: Inbox directory (defaults to folder/user)
        outbox: Outbox directory (defaults to inbox)
        strict: Enable strict mode for serialization
        policy: Security policy for deserialization
        auto_load_replies: Auto-load computation replies (default: True)
        poll_interval: How often to check for replies in seconds (default: 2.0)

        # SyftBox parameters (for encrypted mode):
        syftbox: Enable SyftBox backend for encrypted sync (default: True)
        data_dir: SyftBox data directory (contains datasites/, .syc/)
        vault_path: Override .syc vault location
        disable_crypto: Disable encryption for testing (default: False)

    Returns:
        BeaverContext with auto-loading enabled

    Examples:
        # SyftBox encrypted mode (default)
        bv = beaver.connect(
            user="client1@sandbox.local",
            data_dir=Path.cwd(),
        )

        # Legacy filesystem mode
        bv = beaver.connect("shared", user="bob", syftbox=False)
    """
    import os

    def _env_flag(name: str) -> bool:
        return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}

    # Allow tests/CLI to force local filesystem mode without changing notebook code
    if (
        _env_flag("BEAVER_LOCAL_MODE")
        or _env_flag("BEAVER_DISABLE_SYFTBOX")
        or _env_flag("BEAVER_FORCE_LEGACY")
    ):
        syftbox = False

    # Auto-detect user from environment if not provided
    if user == "unknown":
        user = os.environ.get("BEAVER_USER") or os.environ.get("SYFTBOX_EMAIL") or "unknown"

    # Auto-fallback to legacy mode if caller passed a folder but no data_dir
    if syftbox and data_dir is None and folder is not None:
        syftbox = False

    # Auto-detect data_dir from environment variables
    if syftbox and data_dir is None:
        import os

        data_dir = os.environ.get("SYFTBOX_DATA_DIR") or os.environ.get("BIOVAULT_HOME")

    if syftbox:
        # SyftBox encrypted mode
        if data_dir is None:
            raise ValueError(
                "data_dir is required when syftbox=True (or set SYFTBOX_DATA_DIR env var)"
            )

        # Auto-detect email from SyftBox config if not provided
        effective_user = user
        if user == "unknown":
            import json
            import os

            # 1) Explicit env var
            effective_user = os.environ.get("SYFTBOX_EMAIL", user)

            # 2) Env override for config path
            if effective_user == "unknown":
                env_config = os.environ.get("SYFTBOX_CONFIG_PATH")
                if env_config:
                    config_path = Path(env_config)
                    if config_path.exists():
                        try:
                            config = json.loads(config_path.read_text())
                            effective_user = config.get("email", user)
                        except Exception:
                            pass

            # 3) Config files in standard locations
            if effective_user == "unknown":
                config_candidates = [
                    Path(data_dir) / "syftbox" / "config.json",  # desktop location
                    Path(data_dir) / ".syftbox" / "config.json",  # legacy CLI location
                ]

                for config_path in config_candidates:
                    if config_path.exists():
                        try:
                            config = json.loads(config_path.read_text())
                            effective_user = config.get("email", user)
                            if effective_user != user:
                                break
                        except Exception:
                            continue

        from .syftbox_backend import SyftBoxBackend

        backend = SyftBoxBackend(
            data_dir=data_dir,
            email=effective_user,
            vault_path=vault_path,
            disable_crypto=disable_crypto,
        )

        return BeaverContext(
            inbox=backend.shared_path,
            outbox=backend.shared_path,
            user=effective_user,
            biovault=biovault,
            strict=strict,
            policy=policy,
            auto_load_replies=auto_load_replies,
            poll_interval=poll_interval,
            _backend=backend,
        )
    else:
        # Local filesystem mode
        # Allow env override for folder (shared session dir) even if data_dir was provided
        session_dir_env = (
            os.environ.get("BEAVER_LOCAL_SESSION_DIR")
            or os.environ.get("BEAVER_SESSION_DIR")
            or os.environ.get("BEAVER_LOCAL_FOLDER")
            or os.environ.get("BEAVER_LEGACY_FOLDER")
        )
        if session_dir_env:
            folder = session_dir_env
        elif folder is None and data_dir is not None:
            folder = data_dir

        if folder is None:
            raise ValueError("folder is required when syftbox=False")
        base = Path(folder)

        # Optionally keep all users in the same folder for tests (no per-user subdir)
        shared_local = _env_flag("BEAVER_LOCAL_SHARED") or _env_flag("BEAVER_LEGACY_SHARED")
        user_subdir = base if shared_local else (base / user if user else base)

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
