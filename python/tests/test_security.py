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


def test_envelope_load_blocks_function_payload():
    """Envelope.load should refuse function payloads under default policy."""

    def evil():
        return "bad"

    env = runtime.pack(evil, sender="attacker")
    with pytest.raises(runtime.SecurityError):
        env.load(inject=False)


def test_twin_load_blocks_untrusted_loader_auto_accept(tmp_path):
    """Twin.load should still block untrusted loader even with auto_accept=True."""
    from beaver.twin import Twin

    loader = {
        "_trusted_loader": True,
        "path": str(tmp_path / "artifact.bin"),
        "deserializer_src": "def load(p):\n    return __import__('os').getpid()",
    }
    twin = Twin(public=loader, private=None, owner="attacker")

    with pytest.raises(runtime.SecurityError):
        twin.load(which="public", auto_accept=True)


def test_trusted_loader_allows_allowed_import_and_globals(monkeypatch, tmp_path):
    """Trusted loader path allows allowed imports and globals() without exposing full globals."""
    monkeypatch.setenv("BEAVER_TRUSTED_LOADERS", "1")
    loader = {
        "_trusted_loader": True,
        "path": str(tmp_path / "data.bin"),
        "deserializer_src": (
            "import json\n"
            "def load(p):\n"
            "    g = globals()\n"
            "    return {'p': p, 'mod': json.__name__, 'global_ok': g is not None}\n"
        ),
    }

    result = runtime._resolve_trusted_loader(loader, auto_accept=True, backend=None)
    assert result["p"] == loader["path"]
    assert result["mod"] == "json"
    assert result["global_ok"] is True


def test_trusted_loader_blocks_disallowed_import(monkeypatch, tmp_path):
    """Even in trusted mode, disallowed imports are blocked."""
    monkeypatch.setenv("BEAVER_TRUSTED_LOADERS", "1")
    loader = {
        "_trusted_loader": True,
        "path": str(tmp_path / "data.bin"),
        "deserializer_src": "import os\ndef load(p):\n    return os.getcwd()\n",
    }

    with pytest.raises(runtime.SecurityError):
        runtime._resolve_trusted_loader(loader, auto_accept=True, backend=None)


def test_trusted_policy_allows_function_deserialize(monkeypatch):
    """Trusted policy env should allow function payloads."""

    def f():
        return "ok"

    monkeypatch.setenv("BEAVER_TRUSTED_POLICY", "1")
    env = runtime.pack(f, sender="me")
    # Should not raise
    obj = runtime.unpack(env, auto_accept=True)
    assert callable(obj)


def test_twin_attribute_access_does_not_auto_execute_loader(tmp_path):
    """Accessing Twin.public or Twin.private should NOT auto-execute loader code.

    This tests the __getattribute__ hook security - simply reading the attribute
    should return the raw TrustedLoader dict, not trigger code execution.

    Uses only allowed imports (pathlib) to ensure the security comes from
    NOT auto-executing, rather than from import blocking.
    """
    from beaver.twin import Twin

    marker = tmp_path / "getattr_pwned.txt"
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"dummy data")

    # Use only allowed imports to isolate the auto-execute behavior
    loader_with_side_effect = {
        "_trusted_loader": True,
        "path": str(artifact),
        "deserializer_src": (
            f"import pathlib\n"
            f"def load(p):\n"
            f"    pathlib.Path('{marker}').write_text('executed')\n"
            f"    return 'loaded'\n"
        ),
    }

    twin = Twin(public=loader_with_side_effect, private=None, owner="attacker")

    # Accessing .public should NOT execute the loader - should return raw dict
    result = twin.public

    # The result should be the raw TrustedLoader dict, NOT the executed result
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert result.get("_trusted_loader") is True, "Should return raw TrustedLoader dict"
    assert not marker.exists(), "Loader was auto-executed just by accessing .public!"


def test_twin_private_attribute_access_does_not_auto_execute_loader(tmp_path):
    """Same test for .private attribute access."""
    from beaver.twin import Twin

    marker = tmp_path / "private_getattr_pwned.txt"
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"dummy data")

    # Use only allowed imports
    loader_with_side_effect = {
        "_trusted_loader": True,
        "path": str(artifact),
        "deserializer_src": (
            f"import pathlib\n"
            f"def load(p):\n"
            f"    pathlib.Path('{marker}').write_text('executed')\n"
            f"    return 'loaded'\n"
        ),
    }

    # Need public to be set for Twin to be valid
    twin = Twin(public="safe_mock", private=loader_with_side_effect, owner="attacker")

    # Accessing .private should NOT execute the loader - should return raw dict
    result = twin.private

    # The result should be the raw TrustedLoader dict
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert result.get("_trusted_loader") is True, "Should return raw TrustedLoader dict"
    assert not marker.exists(), "Loader was auto-executed just by accessing .private!"
