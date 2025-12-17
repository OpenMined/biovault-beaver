"""Minimal CLI to load a session, apply .beaver files, and save a snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from . import load, read_envelope, save, snapshot


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Beaver session runner")
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="Existing snapshot/session .beaver to load before applying files",
    )
    parser.add_argument(
        "--apply",
        nargs="*",
        type=str,
        default=[],
        help="One or more .beaver files to apply in order; if an entry is not a file path it will be exec()'d as Python code.",
    )
    parser.add_argument(
        "--exec",
        dest="exec_snippets",
        action="append",
        default=[],
        help="Inline Python code to exec before saving (can be repeated).",
    )
    parser.add_argument(
        "--unsafe-exec",
        action="store_true",
        help="ALLOW executing inline code via --apply/--exec (disabled by default for safety).",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Path to write the resulting snapshot (default: overwrite --snapshot if set)",
    )
    parser.add_argument(
        "--no-inject",
        action="store_true",
        help="Do not inject payloads into globals when loading files",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug info about loaded objects and executed code",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    inject = not args.no_inject
    unsafe = args.unsafe_exec

    if args.snapshot:
        load(args.snapshot, inject=inject)
        if args.debug:
            print(f"[debug] Loaded snapshot: {args.snapshot}")

    for item in args.apply:
        p = Path(item)
        if p.exists():
            env = read_envelope(p)
            obj = load(p, inject=inject)
            if args.debug:
                print(
                    f"[debug] Applied file: {p} id={env.envelope_id} name={env.name} type={type(obj).__name__} manifest={env.manifest}"
                )
        else:
            if not unsafe:
                print(f"[error] Inline code execution is disabled. Refused: {item}")
                return 1
            exec(item, globals())
            if args.debug:
                print(f"[debug] Executed code: {item}")

    for code in args.exec_snippets:
        if not unsafe:
            print(f"[error] Inline code execution is disabled. Refused: {code}")
            return 1
        exec(code, globals())
        if args.debug:
            print(f"[debug] Executed code snippet: {code}")

    to_save = args.save or args.snapshot
    if to_save:
        # If a specific path is requested, use save(); otherwise use snapshot() to auto-name.
        out_path = (
            save(to_save)
            if args.save
            else snapshot(out_dir=to_save.parent if isinstance(to_save, Path) else ".")
        )
        print(f"Saved snapshot: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
