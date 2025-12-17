from __future__ import annotations

import argparse
import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4


def _repo_root(start: Path) -> Path:
    current = start.resolve()
    for _ in range(8):
        if (current / "python" / "src" / "beaver").exists() and (current / "notebooks").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not locate repo root (expected python/src/beaver and notebooks/).")


def _extract_function_from_notebook(notebook_path: Path, func_name: str) -> str:
    nb = json.loads(notebook_path.read_text())
    cells = nb.get("cells", [])
    sources: list[str] = []
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        sources.append(str(src))
    joined = "\n\n".join(sources)

    needle = f"def {func_name}("
    idx = joined.find(needle)
    if idx < 0:
        raise RuntimeError(f"Function '{func_name}' not found in {notebook_path}")

    # Naive slice from the function def until the next top-level def/class.
    tail = joined[idx:]
    lines = tail.splitlines()
    out: list[str] = []
    out.append(lines[0])
    for line in lines[1:]:
        if line.startswith("def ") or line.startswith("class "):
            break
        out.append(line)
    return "\n".join(out).strip() + "\n"


def _build_function(src: str, func_name: str):
    ns: dict = {}
    exec(compile(textwrap.dedent(src), filename="<notebook>", mode="exec"), ns, ns)
    fn = ns.get(func_name)
    if not callable(fn):
        raise RuntimeError(f"Failed to build callable '{func_name}' from extracted source.")
    return fn


def _make_request_object(func, *, sender: str, result_name: str, bad_schema: bool):
    # Import lazily so this script can run from repo without install.
    from beaver.computation import ComputationRequest as GoodRequest

    if not bad_schema:
        return GoodRequest(
            comp_id=uuid4().hex,
            result_id=uuid4().hex,
            func=func,
            args=(
                {
                    "_beaver_remote_var": True,
                    "name": "gwas_data",
                    "owner": "madhava@openmined.org",
                    "var_type": "Twin[dict]",
                },
            ),
            kwargs={},
            sender=sender,
            result_name=result_name,
        )

    # Generate a pathological schema-mismatch payload:
    # Replace beaver.computation.ComputationRequest *on the sender side* with a
    # compatible-looking type name but different dataclass schema.
    import beaver.computation as comp_mod

    original = comp_mod.ComputationRequest

    @dataclass
    class ComputationRequest:  # noqa: N801 - must match receiver name for schema hash mismatch
        comp_id: str
        result_id: str
        func: object
        args: tuple
        kwargs: dict
        sender: str
        result_name: str
        created_at: str = field(default_factory=lambda: "2025-01-01T00:00:00+00:00")
        # Schema drift: extra field receiver doesn't have (or order drift).
        schema_drift: str = "bad"

    try:
        ComputationRequest.__module__ = "beaver.computation"
        ComputationRequest.__qualname__ = "ComputationRequest"
        comp_mod.ComputationRequest = ComputationRequest  # type: ignore[assignment]
        req = comp_mod.ComputationRequest(  # type: ignore[call-arg]
            comp_id=uuid4().hex,
            result_id=uuid4().hex,
            func=func,
            args=(
                {
                    "_beaver_remote_var": True,
                    "name": "gwas_data",
                    "owner": "madhava@openmined.org",
                    "var_type": "Twin[dict]",
                },
            ),
            kwargs={},
            sender=sender,
            result_name=result_name,
        )
    finally:
        comp_mod.ComputationRequest = original  # type: ignore[assignment]

    return req


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate GWAS request .beaver fixtures.")
    parser.add_argument(
        "--notebook",
        type=Path,
        default=None,
        help="Path to GWAS notebook (default: notebooks/05-gwas-do.ipynb)",
    )
    parser.add_argument(
        "--func",
        type=str,
        default="gwas_step1_merge",
        help="Function name to extract from notebook",
    )
    parser.add_argument(
        "--sender",
        type=str,
        default="test@madhavajay.com",
        help="Sender email in the request",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="request_gwas_step1_merge_for_merge_result",
        help="Envelope name",
    )
    parser.add_argument(
        "--result-name",
        type=str,
        default="merge_result",
        help="ComputationRequest.result_name",
    )
    parser.add_argument(
        "--bad-schema",
        action="store_true",
        help="Generate a schema-mismatch (pathological) ComputationRequest payload",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: sandbox/fixtures/gwas_request)",
    )
    args = parser.parse_args(argv)

    root = _repo_root(Path(__file__))
    notebook = args.notebook or (root / "notebooks" / "05-gwas-do.ipynb")
    out_dir = args.out or (root / "sandbox" / "fixtures" / "gwas_request")
    out_dir.mkdir(parents=True, exist_ok=True)

    from beaver.policy import TRUSTED_POLICY
    from beaver.runtime import pack, write_envelope

    src = _extract_function_from_notebook(notebook, args.func)
    fn = _build_function(src, args.func)
    req = _make_request_object(
        fn, sender=args.sender, result_name=args.result_name, bad_schema=args.bad_schema
    )

    env = pack(req, sender=args.sender, name=args.name, policy=TRUSTED_POLICY)
    path = write_envelope(env, out_dir=out_dir)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
