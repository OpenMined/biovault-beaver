import pickle
import types
from pathlib import Path

import pytest

from beaver import cli, runtime
from beaver.envelope import BeaverEnvelope


def test_trusted_loader_exec_is_blocked(tmp_path):
    marker = tmp_path / "loader_pwned.txt"
    payload = {
        "_trusted_loader": True,
        "deserializer_src": (
            "import pathlib\n"
            f"pathlib.Path('{marker}').write_text('pwned')\n"
            "def load(p):\n"
            "    return 'ok'\n"
        ),
        "path": str(tmp_path / "artifact.bin"),
    }

    with pytest.raises(runtime.SecurityError):
        runtime._resolve_trusted_loader(payload, auto_accept=True, backend=None)

    assert not marker.exists()


def test_pickle_fallback_blocks_reduce_execution(tmp_path, monkeypatch):
    marker = tmp_path / "pickle_pwned.txt"

    class Exploit:
        def __reduce__(self):
            return (Path(marker).write_text, ("owned",))

    payload = pickle.dumps(Exploit())
    env = BeaverEnvelope(payload=payload)

    class PickleFory:
        def __init__(self, *args, **kwargs):
            pass

        def register_type(self, *_args, **_kwargs):
            return None

        def dumps(self, obj):
            return pickle.dumps(obj)

        def loads(self, raw):
            return pickle.loads(raw)

    monkeypatch.setattr(runtime, "pyfory", types.SimpleNamespace(Fory=PickleFory))

    with pytest.raises(runtime.SecurityError):
        runtime.unpack(env, auto_accept=True)

    assert not marker.exists()


def test_trusted_loader_path_traversal_blocked(tmp_path):
    traversal_path = tmp_path / ".." / ".." / "escape.bin"
    payload = {
        "_trusted_loader": True,
        "path": str(traversal_path),
        "deserializer_src": (
            "import pathlib\n"
            "def load(p):\n"
            "    # Simulate reading arbitrary path\n"
            "    return pathlib.Path(p).exists()\n"
        ),
    }

    with pytest.raises(runtime.SecurityError):
        runtime._resolve_trusted_loader(payload, auto_accept=True, backend=None)


def test_cli_rejects_inline_exec(tmp_path):
    marker = tmp_path / "cli_pwned.txt"
    code = f"__import__('pathlib').Path('{marker}').write_text('pwned')"

    rc = cli.main(["--apply", code, "--no-inject"])

    assert rc != 0
    assert not marker.exists()


def test_codebase_has_no_pickle_imports():
    """Ensure pickle is not imported anywhere in beaver code."""
    root = Path(__file__).resolve().parents[1] / "src" / "beaver"
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text()
        if "import pickle" in text or "from pickle" in text:
            offenders.append(path)
    assert not offenders, f"pickle imports found: {offenders}"


def test_unpack_blocks_function_without_policy():
    """Unpacking a function payload should be blocked by default-deny policy."""

    def evil():
        return "bad"

    env = runtime.pack(evil, sender="attacker")
    with pytest.raises(runtime.SecurityError):
        runtime.unpack(env)


def test_auto_accept_does_not_run_untrusted_loader():
    payload = {
        "_trusted_loader": True,
        "deserializer_src": "def load(p):\n    return __import__('os').getpid()",
        "path": "/tmp/fake.bin",
    }
    env = BeaverEnvelope(payload=runtime.pack(payload).payload)
    with pytest.raises(runtime.SecurityError):
        runtime.unpack(env, auto_accept=True)
