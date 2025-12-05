"""Tests that beaver core works without lib-support dependencies installed."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def _get_project_pythonpath() -> str:
    """Get the PYTHONPATH for the current project's src directory."""
    tests_dir = Path(__file__).parent
    src_dir = tests_dir.parent / "src"
    return str(src_dir)


def test_beaver_imports_without_lib_support():
    """Verify beaver can be imported in an environment without lib-support deps."""
    script = textwrap.dedent("""
        import sys
        import builtins

        # Block lib-support optional dependencies (numpy is a core dep, not blocked)
        BLOCKED = {'pandas', 'PIL', 'pillow', 'matplotlib', 'torch',
                   'safetensors', 'anndata', 'pyarrow'}

        real_import = builtins.__import__
        def blocking_import(name, *args, **kwargs):
            root = name.split('.')[0].lower()
            if root in BLOCKED:
                raise ImportError(f'Simulated missing: {name}')
            return real_import(name, *args, **kwargs)

        builtins.__import__ = blocking_import

        # These imports should succeed without lib-support deps
        from beaver.runtime import TrustedLoader, pack, unpack
        from beaver.lib_support import register_builtin_loader
        from beaver.computation import ComputationRequest
        from beaver.envelope import BeaverEnvelope

        print("All core imports succeeded")
    """)

    env = os.environ.copy()
    env["PYTHONPATH"] = _get_project_pythonpath()

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, f"Import failed: {result.stderr}"
    assert "All core imports succeeded" in result.stdout


def test_register_builtin_loader_ignores_unknown_types():
    """Verify register_builtin_loader gracefully ignores non-lib-support types."""
    from beaver.lib_support import register_builtin_loader
    from beaver.runtime import TrustedLoader

    # Should not raise for arbitrary objects
    register_builtin_loader("a string", TrustedLoader)
    register_builtin_loader(123, TrustedLoader)
    register_builtin_loader({"a": "dict"}, TrustedLoader)
    register_builtin_loader([1, 2, 3], TrustedLoader)

    # These types should not be registered
    assert TrustedLoader.get(str) is None
    assert TrustedLoader.get(int) is None
    assert TrustedLoader.get(dict) is None
    assert TrustedLoader.get(list) is None


def test_pack_unpack_works_without_lib_support():
    """Verify pack/unpack works for basic types without lib-support libs."""
    script = textwrap.dedent("""
        import sys
        import builtins

        # Block lib-support optional dependencies (numpy is a core dep, not blocked)
        BLOCKED = {'pandas', 'PIL', 'pillow', 'matplotlib', 'torch',
                   'safetensors', 'anndata', 'pyarrow'}

        real_import = builtins.__import__
        def blocking_import(name, *args, **kwargs):
            root = name.split('.')[0].lower()
            if root in BLOCKED:
                raise ImportError(f'Simulated missing: {name}')
            return real_import(name, *args, **kwargs)

        builtins.__import__ = blocking_import

        from beaver.runtime import pack, unpack

        # Pack and unpack a simple dict
        data = {"key": "value", "nums": [1, 2, 3]}
        envelope = pack(data, sender="test", name="test_data")
        result = unpack(envelope)

        assert result == data, f"Expected {data}, got {result}"
        print("pack/unpack works")
    """)

    env = os.environ.copy()
    env["PYTHONPATH"] = _get_project_pythonpath()

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, f"Test failed: {result.stderr}"
    assert "pack/unpack works" in result.stdout
