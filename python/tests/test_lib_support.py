from __future__ import annotations

import contextlib
import inspect
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Callable, Optional

import pytest

from beaver.lib_support import register_builtin_loader
from beaver.runtime import TrustedLoader


def summarize(obj: Any) -> dict[str, Any]:
    import hashlib
    from io import BytesIO

    summary: dict[str, Any] = {"type": f"{type(obj).__module__}.{type(obj).__name__}"}

    try:
        import numpy as np
    except ImportError:  # pragma: no cover - guarded in runtime
        np = None  # type: ignore[assignment]
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - guarded in runtime
        pd = None  # type: ignore[assignment]
    try:
        import torch
    except ImportError:  # pragma: no cover - guarded in runtime
        torch = None  # type: ignore[assignment]
    with contextlib.suppress(ImportError):
        from PIL import Image  # noqa: F401
    try:
        import matplotlib

        matplotlib.use("Agg")
    except Exception:  # pragma: no cover - guarded in runtime
        matplotlib = None  # type: ignore[assignment]

    if np is not None and isinstance(obj, np.ndarray):
        summary.update(
            {
                "kind": "ndarray",
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "data": obj.tolist(),
            }
        )
        return summary

    if torch is not None and isinstance(obj, torch.Tensor):
        arr = obj.detach().cpu().numpy()
        summary.update(
            {
                "kind": "torch_tensor",
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "device": str(obj.device),
                "data": arr.tolist(),
            }
        )
        return summary

    if pd is not None and isinstance(obj, pd.DataFrame):
        summary.update(
            {
                "kind": "dataframe",
                "shape": list(obj.shape),
                "columns": list(obj.columns),
                "dtypes": [str(dt) for dt in obj.dtypes],
                "index": obj.index.tolist(),
                "data": obj.to_dict(orient="list"),
            }
        )
        return summary

    if pd is not None and isinstance(obj, pd.Series):
        summary.update(
            {
                "kind": "series",
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "name": obj.name,
                "index": obj.index.tolist(),
                "data": obj.tolist(),
            }
        )
        return summary

    if pd is not None and isinstance(obj, pd.Index):
        summary.update(
            {
                "kind": "index",
                "dtype": str(obj.dtype),
                "name": obj.name,
                "data": obj.tolist(),
            }
        )
        return summary

    if Image is not None and isinstance(obj, Image.Image):
        arr = [list(pix) for pix in obj.getdata()]
        summary.update(
            {
                "kind": "pillow",
                "mode": obj.mode,
                "size": list(obj.size),
                "data": arr,
            }
        )
        return summary

    if matplotlib is not None and obj.__class__.__module__.startswith("matplotlib.figure"):
        buf = BytesIO()
        obj.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        digest = hashlib.sha256(buf.getvalue()).hexdigest()
        summary.update({"kind": "matplotlib_figure", "sha256": digest})
        return summary

    try:
        import anndata as ad
    except ImportError:  # pragma: no cover - guarded in runtime
        ad = None  # type: ignore[assignment]

    if ad is not None and isinstance(obj, ad.AnnData):
        summary.update(
            {
                "kind": "anndata",
                "shape": list(obj.shape),
                "obs_columns": list(obj.obs.columns),
                "var_columns": list(obj.var.columns),
            }
        )
        return summary

    summary["repr"] = repr(obj)
    return summary


SUMMARY_SOURCE = textwrap.dedent(inspect.getsource(summarize))


def _clean_source(fn: Callable[..., Any]) -> str:
    lines = inspect.getsource(fn).splitlines()
    cleaned = "\n".join(line for line in lines if not line.lstrip().startswith("@"))
    return textwrap.dedent(cleaned)


def _run_deserializer_in_subprocess(
    src: str, data_path: Path, meta: Optional[dict] = None
) -> dict[str, Any]:
    script_lines = [
        "import json",
        "from pathlib import Path",
        "from typing import Any, Dict",
        "import contextlib",
        "",
        "# best-effort imports so deserializer source can reference them without exploding",
        "scope: Dict[str, Any] = {}",
        "scope['Path'] = Path",
        "scope['Any'] = Any",
        "scope['Dict'] = Dict",
        "for _stmt in [",
        '    "import numpy as np",',
        '    "import pandas as pd",',
        '    "from PIL import Image",',
        "    \"import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt\",",
        '    "import torch",',
        '    "from safetensors.torch import load_file, save_file",',
        '    "import anndata as ad",',
        '    "import json",',
        "]:",
        "    try:",
        "        exec(_stmt, scope, scope)",
        "    except Exception:",
        "        pass",
        "",
        # Inject metadata if provided (mirrors runtime injection)
        f"scope['beaver_meta'] = {json.dumps(meta) if meta is not None else 'None'}",
        "",
        SUMMARY_SOURCE,
        f"exec({src!r}, scope, scope)",
        "deser_candidates = [v for v in scope.values() if callable(v) and getattr(v, '__name__', '').endswith('deserialize_file')]",
        "if not deser_candidates:",
        "    deser_candidates = [v for v in scope.values() if callable(v)]",
        "deser_fn = deser_candidates[0]",
        f"obj = deser_fn(Path({json.dumps(str(data_path))}))",
        "print(json.dumps(summarize(obj)))",
    ]
    script = "\n".join(script_lines)
    result = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )
    return json.loads(result.stdout.strip())


def _run_deserializer_with_python(
    python_bin: Path, src: str, data_path: Path, missing_modules: Optional[list[str]] = None
) -> subprocess.CompletedProcess:
    script_lines = [
        "import json",
        "from pathlib import Path",
        "from typing import Any, Dict",
        "import builtins",
        "scope: Dict[str, Any] = {}",
        "scope['Path'] = Path",
        "scope['Any'] = Any",
        "scope['Dict'] = Dict",
        "missing = set(json.loads(" + json.dumps(json.dumps(missing_modules or [])) + "))",
        "real_import = builtins.__import__",
        "def _fake_import(name, *args, **kwargs):",
        "    root = name.split('.')[0]",
        "    if root in missing:",
        "        raise ImportError(f'Blocked import for testing: {root}')",
        "    return real_import(name, *args, **kwargs)",
        "builtins.__import__ = _fake_import",
        f"exec({src!r}, scope, scope)",
        "deser_candidates = [v for v in scope.values() if callable(v) and getattr(v, '__name__', '').endswith('deserialize_file')]",
        "if not deser_candidates:",
        "    deser_candidates = [v for v in scope.values() if callable(v)]",
        "deser_fn = deser_candidates[0]",
        f"deser_fn(Path({json.dumps(str(data_path))}))",
    ]
    script = "\n".join(script_lines)
    return subprocess.run(
        [str(python_bin), "-c", script],
        capture_output=True,
        text=True,
    )


def _numpy_factory():
    np = pytest.importorskip("numpy")
    return np.arange(6, dtype=np.int32).reshape(2, 3)


def _pandas_dataframe_factory():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    return pd.DataFrame({"a": [1, 2], "b": [3.5, 4.5]})


def _pandas_series_factory():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    return pd.Series([10, 20, 30], index=["x", "y", "z"], name="vals")


def _pandas_index_factory():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    return pd.Index([5, 6, 7], name="idx")


def _pillow_image_factory():
    image_mod = pytest.importorskip("PIL.Image")
    img = image_mod.new("RGB", (2, 2), color=(255, 0, 0))
    return img


def _matplotlib_figure_factory():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    plt = pytest.importorskip("matplotlib.pyplot")
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_axis_off()
    return fig


def _torch_tensor_factory():
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    return torch.arange(8, dtype=torch.float32).reshape(2, 4)


def _anndata_factory():
    ad = pytest.importorskip("anndata")
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    data = np.array([[1, 2], [3, 4]], dtype=float)
    obs = pd.DataFrame({"celltype": ["a", "b"]})
    var = pd.DataFrame({"gene": ["g1", "g2"]})
    return ad.AnnData(X=data, obs=obs, var=var)


CASES = [
    {"name": "numpy_array", "factory": _numpy_factory},
    {"name": "pandas_dataframe", "factory": _pandas_dataframe_factory},
    {"name": "pandas_series", "factory": _pandas_series_factory},
    {"name": "pandas_index", "factory": _pandas_index_factory},
    {"name": "pillow_image", "factory": _pillow_image_factory},
    {"name": "matplotlib_figure", "factory": _matplotlib_figure_factory},
    {"name": "torch_tensor", "factory": _torch_tensor_factory},
    {"name": "anndata", "factory": _anndata_factory},
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_roundtrip_serialization(case: dict[str, Any], tmp_path: Path) -> None:
    obj = case["factory"]()
    register_builtin_loader(obj, TrustedLoader)
    handler = TrustedLoader.get(type(obj))
    assert handler is not None, "Handler should be registered for object"

    data_path = tmp_path / f"{case['name']}.bin"
    meta = handler["serializer"](obj, data_path)
    deser_src = _clean_source(handler["deserializer"])

    expected = summarize(obj)
    observed = _run_deserializer_in_subprocess(deser_src, data_path, meta=meta)
    assert expected == observed


def test_missing_dependency_prompt(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    df = _pandas_dataframe_factory()
    register_builtin_loader(df, TrustedLoader)
    handler = TrustedLoader.get(type(df))
    assert handler is not None

    data_path = tmp_path / "pandas_missing.bin"
    handler["serializer"](df, data_path)
    deser_src = _clean_source(handler["deserializer"])

    # Simulate missing dependencies by blocking imports in subprocess
    result = _run_deserializer_with_python(
        Path(sys.executable), deser_src, data_path, missing_modules=["pandas", "pyarrow"]
    )
    assert result.returncode != 0
    combined_output = (result.stdout or "") + (result.stderr or "")
    assert "pandas" in combined_output or "pyarrow" in combined_output
    assert "Install" in combined_output or "lib-support" in combined_output
