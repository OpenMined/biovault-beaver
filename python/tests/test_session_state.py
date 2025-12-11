"""Integration-like tests for session save/load across process restarts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _run(code: str, env: dict) -> subprocess.CompletedProcess:
    """Run python -c with given code and environment."""
    return subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        env=env,
        capture_output=True,
    )


def test_session_save_load_across_processes(tmp_path: Path) -> None:
    """Save state in one process, reload in another, and warn on mismatched session."""
    env = os.environ.copy()
    env["BEAVER_STATE_DIR"] = str(tmp_path)
    session_id = "sess123"
    owner = "alice@example.com"
    peer = "bob@example.com"

    # First process: create variables and save state
    save_script = f"""
from beaver.session import Session
session = Session(session_id="{session_id}", peer="{peer}", owner="{owner}", role="requester")
foo = 42
bar = {{"key": "value"}}
saved = session.save()
print(saved)
"""
    save_proc = _run(save_script, env)
    assert save_proc.returncode == 0, save_proc.stderr
    saved_path = save_proc.stdout.strip().splitlines()[-1]
    assert saved_path

    # Second process: load state and verify variables round-trip
    load_script = f"""
from beaver.session import Session
session = Session(session_id="{session_id}", peer="{peer}", owner="{owner}", role="requester")
session.load(path="{saved_path}", overwrite=True)
print(f"foo={{foo}}")
print(f"bar={{bar}}")
"""
    load_proc = _run(load_script, env)
    assert load_proc.returncode == 0, load_proc.stderr
    out_lines = load_proc.stdout.strip().splitlines()
    assert "foo=42" in out_lines
    assert "bar={'key': 'value'}" in out_lines

    # Third process: try loading with a different session_id to trigger warning
    mismatch_script = f"""
from beaver.session import Session
session = Session(session_id="other456", peer="{peer}", owner="{owner}", role="requester")
session.load(path="{saved_path}", overwrite=True)
"""
    mismatch_proc = _run(mismatch_script, env)
    # Should not fail, but should emit a warning about session_id mismatch
    assert mismatch_proc.returncode == 0, mismatch_proc.stderr
    assert "does not match active session other456" in mismatch_proc.stdout
