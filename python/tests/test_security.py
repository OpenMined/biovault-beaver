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


def test_restricted_loader_blocks_underscore_prefixed_variables(tmp_path):
    """RestrictedPython blocks variable names starting with underscore.

    This is a security feature that prevents access to Python internals like
    __class__, __dict__, etc. The deserializer code in lib_support.py must
    avoid using underscore-prefixed variable names or RestrictedPython will
    reject the code entirely.
    """
    loader = {
        "_trusted_loader": True,
        "path": str(tmp_path / "data.bin"),
        "deserializer_src": (
            "from pathlib import Path as _Path\n"
            "_my_var = 'test'\n"
            "def load(p):\n"
            "    return _my_var\n"
        ),
    }

    with pytest.raises(runtime.SecurityError) as exc_info:
        runtime._resolve_trusted_loader(loader, auto_accept=True, backend=None)

    assert (
        "invalid variable name" in str(exc_info.value).lower()
        or "execution blocked" in str(exc_info.value).lower()
    )


def test_lib_support_deserializers_no_underscore_variables():
    """Ensure lib_support.py deserializers don't use underscore-prefixed variable names.

    RestrictedPython blocks variable assignments starting with underscore for security.
    This test scans the lib_support.py source to catch any regressions.
    """
    import ast

    lib_support_path = Path(__file__).resolve().parents[1] / "src" / "beaver" / "lib_support.py"
    source = lib_support_path.read_text()

    # Find all function definitions that look like deserializers
    # (functions containing 'deserialize' in name or registered as TrustedLoader)
    tree = ast.parse(source)

    underscore_vars = []
    for node in ast.walk(tree):
        # Check Name nodes in Store context (variable assignments)
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Store)
            and node.id.startswith("_")
            and not node.id.startswith("__")
        ):
            underscore_vars.append(node.id)
        # Check imports with underscore aliases: "from x import Y as _Z"
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.asname and alias.asname.startswith("_"):
                    underscore_vars.append(f"import alias: {alias.asname}")

    # Filter to only report unique issues
    underscore_vars = list(set(underscore_vars))

    # Allow internal implementation details (class names, private methods)
    # but flag anything that would appear in deserializer function bodies
    forbidden = [v for v in underscore_vars if v not in {"_LazyLoaderSpec", "_SPECS"}]

    assert not forbidden, (
        f"lib_support.py contains underscore-prefixed variables that will break "
        f"RestrictedPython: {forbidden}. Rename to remove underscore prefix."
    )


def test_data_location_path_traversal_attack_blocked():
    """data_location in remote_vars must reject absolute paths and path traversal.

    Attack scenario: A malicious sender could craft a remote_vars.json with
    data_location pointing to sensitive files like ~/.ssh/id_rsa or /etc/passwd.
    When the receiver's code tries to "load" this var, it could read arbitrary files.
    """
    from beaver.remote_vars import DataLocationSecurityError, _sanitize_data_location

    # Attack 1: Absolute Unix path to steal SSH keys
    with pytest.raises(DataLocationSecurityError):
        _sanitize_data_location("/Users/victim/.ssh/id_rsa", base_dir=Path("/some/session"))

    # Attack 2: Absolute Windows path
    with pytest.raises(DataLocationSecurityError):
        _sanitize_data_location("C:\\Users\\victim\\secrets.txt", base_dir=Path("/some/session"))

    # Attack 3: Path traversal to escape session directory
    with pytest.raises(DataLocationSecurityError):
        _sanitize_data_location("../../../etc/passwd", base_dir=Path("/some/session"))

    # Attack 4: Windows-style path traversal
    with pytest.raises(DataLocationSecurityError):
        _sanitize_data_location(
            "..\\..\\..\\Windows\\System32\\config\\SAM", base_dir=Path("/some/session")
        )

    # Attack 5: Encoded traversal attempts
    with pytest.raises(DataLocationSecurityError):
        _sanitize_data_location("data/..%2F..%2Fetc/passwd", base_dir=Path("/some/session"))

    # Attack 6: Null byte injection (could truncate path in some systems)
    with pytest.raises(DataLocationSecurityError):
        _sanitize_data_location("data/safe.beaver\x00/etc/passwd", base_dir=Path("/some/session"))

    # Attack 7: Encoded null byte injection
    with pytest.raises(DataLocationSecurityError):
        _sanitize_data_location("data/safe.beaver%00/etc/passwd", base_dir=Path("/some/session"))


def test_data_location_valid_relative_paths():
    """Valid relative paths should be accepted and normalized to Unix-style."""
    from beaver.remote_vars import _sanitize_data_location

    base = Path("/sessions/abc123")

    # Simple filename
    result = _sanitize_data_location("file.beaver", base_dir=base)
    assert result == "file.beaver"

    # Path in data subdirectory
    result = _sanitize_data_location("data/4c1b51ae.beaver", base_dir=base)
    assert result == "data/4c1b51ae.beaver"

    # Windows-style separators should be normalized to Unix
    result = _sanitize_data_location("data\\subdir\\file.beaver", base_dir=base)
    assert result == "data/subdir/file.beaver"


def test_data_location_syft_urls():
    """syft:// data_location entries should normalize and stay within the session base."""
    from beaver.remote_vars import DataLocationSecurityError, _sanitize_data_location

    base = Path("/tmp/datasites/alice@example.com/shared/biovault/sessions/abc123")
    url = "syft://alice@example.com/shared/biovault/sessions/abc123/data/file.beaver"
    assert _sanitize_data_location(url, base_dir=base) == url

    # Reject traversal inside syft path
    with pytest.raises(DataLocationSecurityError):
        _sanitize_data_location(
            "syft://alice@example.com/shared/biovault/sessions/abc123/../secrets.bin",
            base_dir=base,
        )

    # Reject cross-owner paths
    with pytest.raises(DataLocationSecurityError):
        _sanitize_data_location(
            "syft://bob@example.com/shared/biovault/sessions/abc123/data/file.beaver",
            base_dir=base,
        )


def test_data_location_resolves_correctly_on_read(tmp_path):
    """When reading, relative data_location should resolve against session dir."""
    from beaver.remote_vars import _resolve_data_location

    session_dir = (
        tmp_path / "datasites" / "me@example.com" / "shared" / "biovault" / "sessions" / "abc123"
    )
    relative = "data/4c1b51ae.beaver"

    result = _resolve_data_location(relative, session_dir=session_dir)
    expected = session_dir / "data" / "4c1b51ae.beaver"
    assert result == expected

    syft_url = "syft://me@example.com/shared/biovault/sessions/abc123/data/4c1b51ae.beaver"
    result = _resolve_data_location(syft_url, session_dir=session_dir)
    assert result == expected


def test_trusted_loader_translates_windows_datasites_path(tmp_path):
    """TrustedLoader artifact paths should translate from sender to receiver view."""

    class DummyBackend:
        def __init__(self, data_dir: Path):
            self.data_dir = data_dir
            self.uses_crypto = False

    backend = DummyBackend(tmp_path)
    loader = {
        "_trusted_loader": True,
        "path": (
            "C:\\Users\\azureuser\\Desktop\\BioVault\\datasites\\me@example.com\\shared\\biovault\\"
            "sessions\\abc123\\data\\stocks_public.bin"
        ),
        "deserializer_src": "def load(p):\n    return p\n",
        "name": "pandas.core.frame.DataFrame",
    }

    resolved = runtime._resolve_trusted_loader(
        loader, auto_accept=True, backend=backend, trust_loader=False
    )
    assert (
        str(resolved)
        .replace("\\", "/")
        .endswith(
            "/datasites/me@example.com/shared/biovault/sessions/abc123/data/stocks_public.bin"
        )
    )


def test_trusted_loader_paths_are_posix_and_relative_with_artifact_dir(tmp_path):
    """TrustedLoader artifact paths should be unix-style and relative when artifact_dir is provided."""
    np = pytest.importorskip("numpy")
    arr = np.array([1, 2, 3])
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    loader = runtime._prepare_for_sending(arr, artifact_dir=data_dir, name_hint="arr")  # type: ignore[attr-defined]
    assert isinstance(loader, dict) and loader.get("_trusted_loader") is True
    path = loader.get("path")
    assert isinstance(path, str)
    assert "\\" not in path
    assert not path.startswith("/")
    assert ":" not in path
    assert path.startswith("data/")


def test_trusted_loader_relative_path_resolves_against_envelope_path(tmp_path):
    """Relative TrustedLoader paths should resolve against the source envelope directory."""
    session_dir = tmp_path / "sessions" / "abc123" / "data"
    session_dir.mkdir(parents=True, exist_ok=True)
    artifact = session_dir / "artifact.bin"
    artifact.write_bytes(b"ok")
    env_path = session_dir / "env.beaver"
    env_path.write_text("{}")

    loader = {
        "_trusted_loader": True,
        "path": "data/artifact.bin",
        "deserializer_src": "def load(p):\n    return p\n",
        "name": "test.loader",
    }
    resolved = runtime._resolve_trusted_loader(
        loader, auto_accept=True, backend=None, trust_loader=False, envelope_path=env_path
    )
    assert str(resolved) == str(artifact)


def test_twin_load_uses_source_path_for_relative_artifacts(tmp_path):
    """Twin.load should resolve relative TrustedLoader paths using Twin._source_path."""
    from beaver.twin import Twin

    session_dir = tmp_path / "sessions" / "abc123"
    data_dir = session_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    artifact = data_dir / "artifact.bin"
    artifact.write_bytes(b"ok")

    env_path = data_dir / "env.beaver"
    env_path.write_text("{}")

    loader = {
        "_trusted_loader": True,
        "path": "data/artifact.bin",
        "deserializer_src": "def load(p):\n    return p\n",
        "name": "test.loader",
    }
    twin = Twin(public=loader, private=None, owner="me")
    twin._source_path = str(env_path)
    resolved = twin.load(which="public", auto_accept=True, trust_loader=False)
    assert str(resolved) == str(artifact)


def test_sender_verification_rejects_spoofed_sender(tmp_path):
    """read_envelope_verified should reject envelopes with spoofed sender."""
    try:
        import syftbox_sdk as syft
        from syftbox_sdk import SyftBoxStorage

        if not hasattr(SyftBoxStorage, "read_with_shadow_metadata"):
            pytest.skip("syftbox-sdk missing read_with_shadow_metadata")
    except ImportError:
        pytest.skip("syftbox-sdk not available")

    from beaver.envelope import BeaverEnvelope
    from beaver.syftbox_backend import (
        SenderVerificationError,
        SyftBoxBackend,
    )

    # Set up two identities
    alice_data = tmp_path / "alice"
    bob_data = tmp_path / "bob"

    alice_vault = alice_data / ".syc"
    bob_vault = bob_data / ".syc"

    alice_datasites = alice_data / "datasites"
    bob_datasites = bob_data / "datasites"

    alice_datasites.mkdir(parents=True)
    bob_datasites.mkdir(parents=True)

    # Provision identities
    alice = syft.provision_identity(
        identity="alice@test.local",
        data_root=str(alice_data),
        vault_override=str(alice_vault),
    )
    bob = syft.provision_identity(
        identity="bob@test.local",
        data_root=str(bob_data),
        vault_override=str(bob_vault),
    )

    # Exchange keys
    syft.import_bundle(bob.public_bundle_path, str(alice_vault), "bob@test.local")
    syft.import_bundle(alice.public_bundle_path, str(bob_vault), "alice@test.local")

    # Create Alice's backend
    alice_backend = SyftBoxBackend(
        data_dir=alice_data,
        email="alice@test.local",
        vault_path=alice_vault,
    )

    # Create an envelope that CLAIMS to be from "evil@attacker.com" but is
    # actually signed/encrypted by alice@test.local
    spoofed_envelope = BeaverEnvelope(
        sender="evil@attacker.com",  # Spoofed sender!
        payload=b"malicious payload",
        name="spoofed_message",
    )

    # Write it using Alice's credentials (so crypto signature is from alice)
    test_file = alice_datasites / "alice@test.local" / "shared" / "biovault" / "spoofed.beaver"
    test_file.parent.mkdir(parents=True, exist_ok=True)

    alice_backend.write_envelope(spoofed_envelope, recipients=["bob@test.local"])

    # Copy to Bob's view (simulating sync)
    bob_alice_shared = bob_datasites / "alice@test.local" / "shared" / "biovault"
    bob_alice_shared.mkdir(parents=True, exist_ok=True)
    (bob_alice_shared / "spoofed.beaver").write_bytes(
        (
            alice_datasites
            / "alice@test.local"
            / "shared"
            / "biovault"
            / f"{spoofed_envelope.envelope_id}.beaver"
        ).read_bytes()
    )

    # Create Bob's backend
    bob_backend = SyftBoxBackend(
        data_dir=bob_data,
        email="bob@test.local",
        vault_path=bob_vault,
    )

    # Reading with verification should REJECT the spoofed sender
    with pytest.raises(SenderVerificationError) as exc_info:
        bob_backend.read_envelope_verified(bob_alice_shared / "spoofed.beaver")

    assert exc_info.value.claimed_sender == "evil@attacker.com"
    assert exc_info.value.verified_sender == "alice@test.local"

    # Reading without verification should return the envelope with verified sender info
    envelope, verified_sender, fingerprint = bob_backend.read_envelope_verified(
        bob_alice_shared / "spoofed.beaver", verify=False
    )
    assert envelope.sender == "evil@attacker.com"  # Claimed sender
    assert verified_sender == "alice@test.local"  # Actual sender
    assert len(fingerprint) == 64  # SHA256 hex
