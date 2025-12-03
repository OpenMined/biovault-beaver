"""Unit tests for import detection in @bv decorated functions."""

from beaver.computation import (
    _check_missing_imports,
    _check_missing_imports_with_versions,
    _detect_function_imports,
    _detect_function_imports_with_versions,
)

# Define test functions at module level so inspect.getsource() works


def func_with_simple_import():
    import json

    return json.dumps({})


def func_with_from_import():
    from pathlib import Path

    return Path(".")


def func_with_alias():
    import collections as col

    return col.Counter()


def func_with_submodule():
    import os.path

    return os.path.exists(".")


def func_with_multiple():
    import json
    import sys
    from pathlib import Path

    return sys.version, json.dumps({}), Path(".")


def func_without_imports(x):
    return x + 1


def func_with_stdlib():
    import json

    return json.dumps({})


def func_with_fake_import():
    import nonexistent_package_12345

    return nonexistent_package_12345.foo()


def func_with_sklearn():
    import sklearn

    return sklearn.__version__


def make_violin(adata):
    import matplotlib.pyplot as plt
    import scanpy as sc

    sc.pl.violin(adata, ["n_genes_by_counts"])
    return plt.gcf()


# Tests


def test_detect_simple_import():
    """Test detection of simple import statement."""
    imports = _detect_function_imports(func_with_simple_import)
    assert "json" in imports


def test_detect_from_import():
    """Test detection of from...import statement."""
    imports = _detect_function_imports(func_with_from_import)
    assert "pathlib" in imports


def test_detect_aliased_import():
    """Test detection of aliased import."""
    imports = _detect_function_imports(func_with_alias)
    assert "collections" in imports


def test_detect_submodule_import():
    """Test detection of submodule import returns top-level."""
    imports = _detect_function_imports(func_with_submodule)
    assert "os" in imports


def test_detect_multiple_imports():
    """Test detection of multiple imports."""
    imports = _detect_function_imports(func_with_multiple)
    assert "sys" in imports
    assert "json" in imports
    assert "pathlib" in imports


def test_no_imports():
    """Test function with no imports."""
    imports = _detect_function_imports(func_without_imports)
    assert imports == []


def test_check_missing_imports_stdlib():
    """Test that stdlib imports are not flagged as missing."""
    missing = _check_missing_imports(func_with_stdlib)
    assert missing == []


def test_check_missing_imports_nonexistent():
    """Test that nonexistent packages are flagged."""
    missing = _check_missing_imports(func_with_fake_import)
    assert "nonexistent_package_12345" in missing


def test_package_name_mapping():
    """Test that common package name mappings work."""
    # sklearn maps to scikit-learn
    missing = _check_missing_imports(func_with_sklearn)
    # If sklearn is not installed, it should suggest scikit-learn
    if missing:
        assert "scikit-learn" in missing


def test_detect_scanpy_matplotlib():
    """Test realistic single-cell analysis function."""
    imports = _detect_function_imports(make_violin)
    assert "matplotlib" in imports
    assert "scanpy" in imports


def test_detect_imports_with_versions():
    """Test that version detection works for installed stdlib packages."""
    versions = _detect_function_imports_with_versions(func_with_stdlib)
    # json is stdlib, may not have version metadata (None)
    assert "json" in versions


def test_check_missing_with_versions():
    """Test missing import detection with version info."""
    missing = _check_missing_imports_with_versions(func_with_fake_import, None)
    assert len(missing) == 1
    pkg_name, version = missing[0]
    assert pkg_name == "nonexistent_package_12345"
    assert version is None


def test_check_missing_with_provided_versions():
    """Test that provided versions are included in missing list."""
    required = {"nonexistent_package_12345": "1.2.3"}
    missing = _check_missing_imports_with_versions(func_with_fake_import, required)
    assert len(missing) == 1
    pkg_name, version = missing[0]
    assert pkg_name == "nonexistent_package_12345"
    assert version == "1.2.3"


# Tests for envelope integration


def test_pack_computation_request_includes_versions():
    """Test that pack() stores required_versions in manifest for ComputationRequest."""
    from beaver.computation import ComputationRequest
    from beaver.runtime import pack

    # Create a ComputationRequest with a function that has imports
    request = ComputationRequest(
        comp_id="test123",
        result_id="result456",
        func=func_with_stdlib,
        args=(),
        kwargs={},
        sender="test_user",
        result_name="test_result",
    )

    env = pack(request, sender="test_user", name="test_request")

    # Check that manifest contains required_versions
    assert "required_versions" in env.manifest
    assert "json" in env.manifest["required_versions"]


def test_envelope_check_missing_imports_all_installed():
    """Test that envelope check passes when all imports are installed."""
    from beaver.envelope import BeaverEnvelope

    env = BeaverEnvelope(
        manifest={"required_versions": {"json": None, "os": None}},  # stdlib, installed
    )

    # Should not raise
    env._check_and_prompt_missing_imports()


def test_envelope_check_missing_imports_detects_missing():
    """Test that envelope check detects missing packages."""
    from unittest.mock import patch

    from beaver.envelope import BeaverEnvelope

    env = BeaverEnvelope(
        name="test_func",
        manifest={"required_versions": {"nonexistent_pkg_xyz": "1.0.0"}},
    )

    # Mock the prompt to return False (user declines)
    with patch("beaver.computation._prompt_install_function_deps", return_value=False):
        import pytest

        with pytest.raises(ImportError) as exc_info:
            env._check_and_prompt_missing_imports()
        assert "nonexistent_pkg_xyz" in str(exc_info.value)
        assert "1.0.0" in str(exc_info.value)
