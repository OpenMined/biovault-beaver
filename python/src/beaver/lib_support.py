from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class _LazyLoaderSpec:
    """Lazy registration spec for a library type."""

    name: str
    matcher: Callable[[Any], bool]
    registrar: Callable[[Any, type[Any]], None]
    registered_types: set[type[Any]] = field(default_factory=set)

    def maybe_register(self, obj: Any, trusted_loader_cls: type[Any]) -> None:
        obj_type = type(obj)
        if obj_type in self.registered_types:
            return
        if self.matcher(obj):
            self.registrar(obj, trusted_loader_cls)
            self.registered_types.add(obj_type)


def _match_numpy(obj: Any) -> bool:
    return (
        getattr(obj.__class__, "__module__", "").startswith("numpy")
        and getattr(obj.__class__, "__name__", "") == "ndarray"
    )


def _require(module_name: str, *, feature: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - runtime enforcement
        raise ImportError(
            f"{module_name} is required for {feature}. "
            'Install with `pip install "biovault-beaver[lib-support]"`.'
        ) from exc


def _register_numpy(obj: Any, trusted_loader_cls: type[Any], obj_type: type[Any] = None) -> None:
    np = _require("numpy", feature="numpy array serialization")

    if obj_type is None:
        obj_type = type(obj)
    name = f"{obj_type.__module__}.{obj_type.__name__}"

    @trusted_loader_cls.register(obj_type, name=name)
    def numpy_serialize_file(arr: Any, path: Path) -> None:
        with Path(path).open("wb") as f:
            np.save(f, arr, allow_pickle=False)

    @trusted_loader_cls.register(obj_type, name=name)
    def numpy_deserialize_file(path: Path) -> Any:
        try:
            import numpy as np_local  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime env
            raise ImportError(
                "numpy is required to deserialize numpy arrays. "
                'Install with `pip install "biovault-beaver[lib-support]"`.'
            ) from exc
        with Path(path).open("rb") as f:
            return np_local.load(f, allow_pickle=False)


def _match_pandas(obj: Any) -> bool:
    return getattr(obj.__class__, "__module__", "").startswith("pandas")


def _register_pandas(obj: Any, trusted_loader_cls: type[Any], obj_type: type[Any] = None) -> None:
    pd = _require("pandas", feature="pandas serialization")

    if obj_type is None:
        obj_type = type(obj)
    name = f"{obj_type.__module__}.{obj_type.__name__}"

    @trusted_loader_cls.register(obj_type, name=name)
    def pandas_serialize_file(frame_like: Any, path: Path) -> dict:
        """Serialize pandas object to parquet. Returns metadata dict."""
        _require("pyarrow", feature="pandas parquet serialization")
        from pathlib import Path as PathCls

        path = PathCls(path)
        if isinstance(frame_like, pd.DataFrame):
            meta = {"kind": "dataframe"}
            frame_like.to_parquet(path, index=True, engine="pyarrow")
        elif isinstance(frame_like, pd.Series):
            meta = {"kind": "series", "name": frame_like.name}
            frame_like.to_frame().to_parquet(path, index=True, engine="pyarrow")
        elif isinstance(frame_like, pd.Index):
            meta = {"kind": "index", "name": frame_like.name}
            pd.DataFrame({"__index_values": frame_like}).to_parquet(
                path, index=False, engine="pyarrow"
            )
        else:
            raise TypeError(f"Unsupported pandas type: {type(frame_like)}")
        return meta  # Return metadata to be embedded in TrustedLoader dict

    @trusted_loader_cls.register(obj_type, name=name)
    def pandas_deserialize_file(path: Path, meta: dict = None) -> Any:
        """Deserialize pandas object from parquet. Meta can be passed or read from .meta.json."""
        import io as io_local

        try:
            import pandas as pd_local  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime env
            raise ImportError(
                "pandas is required to deserialize pandas objects. "
                'Install with `pip install "biovault-beaver[lib-support]"`.'
            ) from exc
        try:
            import pyarrow  # noqa: F401  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime env
            raise ImportError(
                "pyarrow is required to deserialize pandas parquet payloads. "
                'Install with `pip install "biovault-beaver[lib-support]"`.'
            ) from exc
        # Use Path from globals (injected by runtime) to avoid RestrictedPython import issues
        # Fall back to import if not available (e.g., direct function call outside runtime)
        Path = globals().get("Path")  # noqa: N806
        if Path is None:
            from pathlib import Path
        path = Path(path)

        # If meta not passed, try to get from beaver_meta (injected) or .meta.json file
        if meta is None:
            injected_meta = globals().get("beaver_meta")
            if injected_meta is not None:
                meta = injected_meta
            else:
                # Fallback: read from .meta.json file (legacy support)
                import json as json_local

                meta_path = path.with_suffix(path.suffix + ".meta.json")
                read_text_fn = globals().get("beaver_read_text")
                if read_text_fn is not None:
                    meta_text = read_text_fn(str(meta_path))
                else:
                    meta_text = meta_path.read_text()
                meta = json_local.loads(meta_text)

        # Read parquet data (may be encrypted)
        read_bytes_fn = globals().get("beaver_read_bytes")
        if read_bytes_fn is not None:
            parquet_bytes = read_bytes_fn(str(path))
            parquet_buffer = io_local.BytesIO(parquet_bytes)
        else:
            parquet_buffer = path

        kind = meta.get("kind")
        if kind == "dataframe":
            return pd_local.read_parquet(parquet_buffer, engine="pyarrow")
        if kind == "series":
            df = pd_local.read_parquet(parquet_buffer, engine="pyarrow")
            ser = df.iloc[:, 0]
            ser.index = df.index
            ser.name = meta.get("name")
            return ser
        if kind == "index":
            df = pd_local.read_parquet(parquet_buffer, engine="pyarrow")
            ser = df["__index_values"]
            return pd_local.Index(ser, name=meta.get("name"))
        raise ValueError(f"Unknown pandas kind: {kind}")


def _match_pillow(obj: Any) -> bool:
    return getattr(obj.__class__, "__module__", "").startswith("PIL.")


def _register_pillow(obj: Any, trusted_loader_cls: type[Any], obj_type: type[Any] = None) -> None:
    from PIL import Image  # type: ignore

    _require("PIL", feature="Pillow image serialization")

    if obj_type is None:
        obj_type = type(obj)
    name = f"{obj_type.__module__}.{obj_type.__name__}"

    @trusted_loader_cls.register(obj_type, name=name)
    def pillow_serialize_file(image: Image.Image, path: Path) -> None:  # type: ignore[name-defined]
        image.save(path, format="PNG")

    @trusted_loader_cls.register(obj_type, name=name)
    def pillow_deserialize_file(path: Path) -> Any:
        try:
            from PIL import Image as Image_local  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime env
            raise ImportError(
                "Pillow is required to deserialize images. "
                'Install with `pip install "biovault-beaver[lib-support]"`.'
            ) from exc
        with Image_local.open(path) as img:  # type: ignore[attr-defined]
            return img.copy()


def _match_matplotlib(obj: Any) -> bool:
    return getattr(obj.__class__, "__module__", "").startswith("matplotlib.figure")


def _register_matplotlib(
    obj: Any, trusted_loader_cls: type[Any], obj_type: type[Any] = None
) -> None:
    import matplotlib

    if obj_type is None:
        obj_type = type(obj)
    name = f"{obj_type.__module__}.{obj_type.__name__}"

    @trusted_loader_cls.register(obj_type, name=name)
    def matplotlib_serialize_file(fig: Any, path: Path) -> None:
        fig.savefig(path, format="png", bbox_inches="tight", pad_inches=0)

    @trusted_loader_cls.register(obj_type, name=name)
    def matplotlib_deserialize_file(path: Path) -> Any:
        import matplotlib.image as mpimg

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        img = mpimg.imread(path)
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.axis("off")
        ax.set_position([0, 0, 1, 1])
        height, width = img.shape[:2]
        fig.set_size_inches(width / fig.get_dpi(), height / fig.get_dpi())
        return fig


def _match_torch(obj: Any) -> bool:
    return (
        getattr(obj.__class__, "__module__", "").startswith("torch")
        and getattr(obj.__class__, "__name__", "") == "Tensor"
    )


def _register_torch(obj: Any, trusted_loader_cls: type[Any], obj_type: type[Any] = None) -> None:
    from safetensors.torch import save_file

    if obj_type is None:
        obj_type = type(obj)
    name = f"{obj_type.__module__}.{obj_type.__name__}"

    @trusted_loader_cls.register(obj_type, name=name)
    def torch_serialize_file(tensor: Any, path: Path) -> None:
        path = Path(path)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        tensor_cpu = tensor.detach().cpu()
        save_file({"tensor": tensor_cpu}, str(path))
        meta = {"device": str(tensor.device), "dtype": str(tensor.dtype)}
        meta_path.write_text(json.dumps(meta))

    @trusted_loader_cls.register(obj_type, name=name)
    def torch_deserialize_file(path: Path) -> Any:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - runtime env
            raise ImportError(
                "torch is required to deserialize torch tensors. "
                'Install with `pip install "biovault-beaver[lib-support]"`.'
            ) from exc
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - runtime env
            raise ImportError(
                "safetensors is required to deserialize torch tensors. "
                'Install with `pip install "biovault-beaver[lib-support]"`.'
            ) from exc

        path = Path(path)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        tensor = load_file(str(path))["tensor"]
        device = meta.get("device")
        if device and device != "cpu":
            try:
                target = torch.device(device)
                if target.type == "cuda" and torch.cuda.is_available():
                    tensor = tensor.to(target)
            except Exception:
                pass
        return tensor


def _match_anndata(obj: Any) -> bool:
    return getattr(obj.__class__, "__module__", "").startswith("anndata")


def _register_anndata(obj: Any, trusted_loader_cls: type[Any], obj_type: type[Any] = None) -> None:
    if obj_type is None:
        obj_type = type(obj)
    name = f"{obj_type.__module__}.{obj_type.__name__}"

    @trusted_loader_cls.register(obj_type, name=name)
    def annadata_serialize_file(adata: Any, path: Path) -> None:
        adata.write_h5ad(path)

    @trusted_loader_cls.register(obj_type, name=name)
    def annadata_deserialize_file(path: Path) -> Any:
        try:
            import anndata as ad_local  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime env
            raise ImportError(
                "anndata is required to deserialize AnnData objects. "
                'Install with `pip install "biovault-beaver[lib-support]"`.'
            ) from exc
        return ad_local.read_h5ad(path)


_SPECS: list[_LazyLoaderSpec] = [
    _LazyLoaderSpec("numpy.ndarray", _match_numpy, _register_numpy),
    _LazyLoaderSpec("pandas", _match_pandas, _register_pandas),
    _LazyLoaderSpec("pillow.Image", _match_pillow, _register_pillow),
    _LazyLoaderSpec("matplotlib.Figure", _match_matplotlib, _register_matplotlib),
    _LazyLoaderSpec("torch.Tensor", _match_torch, _register_torch),
    _LazyLoaderSpec("anndata.AnnData", _match_anndata, _register_anndata),
]


def register_builtin_loader(obj: Any, trusted_loader_cls: type[Any]) -> None:
    """
    Lazily register a TrustedLoader handler for a known library object.

    Args:
        obj: Object being inspected for serialization.
        trusted_loader_cls: TrustedLoader class used for registration.
    """
    for spec in _SPECS:
        spec.maybe_register(obj, trusted_loader_cls)


def register_by_type(typ: type[Any], trusted_loader_cls: type[Any]) -> bool:
    """
    Register a TrustedLoader handler by type (without needing an instance).

    Args:
        typ: The type to register a handler for.
        trusted_loader_cls: TrustedLoader class used for registration.

    Returns:
        True if a handler was registered, False otherwise.
    """
    type_module = getattr(typ, "__module__", "")
    type_name = getattr(typ, "__name__", "")

    for spec in _SPECS:
        if typ in spec.registered_types:
            return True  # Already registered

        # Match by module pattern
        matched = False
        if (
            (
                spec.name == "numpy.ndarray"
                and type_module.startswith("numpy")
                and type_name == "ndarray"
            )
            or spec.name == "pandas"
            and type_module.startswith("pandas")
            or spec.name == "anndata.AnnData"
            and type_module.startswith("anndata")
            or (
                spec.name == "torch.Tensor"
                and type_module.startswith("torch")
                and type_name == "Tensor"
            )
            or spec.name == "pillow.Image"
            and type_module.startswith("PIL")
            or spec.name == "matplotlib.Figure"
            and type_module.startswith("matplotlib.figure")
        ):
            matched = True

        if matched:
            # Call registrar with the type passed explicitly
            spec.registrar(None, trusted_loader_cls, obj_type=typ)
            spec.registered_types.add(typ)
            return True

    return False


__all__ = ["register_builtin_loader", "register_by_type"]
