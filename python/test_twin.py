"""Test Twin integration with beaver."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from beaver import Twin, connect

def test_twin_basic():
    """Test basic Twin creation and usage."""
    print("=" * 60)
    print("Test 1: Basic Twin Creation")
    print("=" * 60)

    # Create a twin with private and public data
    twin = Twin(
        private="secret_data.csv",
        public="mock_data.csv",
        owner="alice",
        name="dataset"
    )

    print(twin)
    print()

    # Test properties
    assert twin.has_private
    assert twin.has_public
    assert twin.value == "secret_data.csv"  # Prefers private
    print("✓ Basic Twin properties work")
    print()


def test_twin_from_mock():
    """Test auto-mock generation."""
    print("=" * 60)
    print("Test 2: Auto-Mock Generation")
    print("=" * 60)

    # Create twin with auto-generated mock
    twin = Twin.from_mock(
        private="real_data.csv",
        owner="bob",
        name="datafile"
    )

    print(twin)
    print()

    assert twin.private == "real_data.csv"
    assert twin.public == "real_data.mock.csv"  # Auto-generated
    print("✓ Auto-mock generation works")
    print()


def test_twin_serialization():
    """Test that beaver's pack/unpack strips private data."""
    print("=" * 60)
    print("Test 3: Privacy-Preserving Serialization")
    print("=" * 60)

    from beaver import pack, unpack

    # Create twin with both sides
    original = Twin(
        private="PRIVATE_SECRET",
        public="PUBLIC_MOCK",
        owner="alice",
        name="sensitive"
    )

    print("Original Twin:")
    print(original)
    print()

    # Pack and unpack using beaver
    envelope = pack(original, sender="alice")
    restored = unpack(envelope)

    print("After pack/unpack:")
    print(restored)
    print()

    # Private should be stripped
    assert not restored.has_private
    assert restored.has_public
    assert restored.public == "PUBLIC_MOCK"
    assert restored.owner == "alice"

    print("✓ Private data stripped during pack/unpack")
    print()


def test_twin_remote_vars():
    """Test Twin with remote vars."""
    print("=" * 60)
    print("Test 4: Twin with Remote Vars")
    print("=" * 60)

    # Create context
    bv = connect("shared", user="alice")

    # Create a twin
    twin = Twin(
        private=[1, 2, 3, 4, 5],
        public=[0, 0, 0],
        owner="alice",
        name="numbers"
    )

    print("Original twin:")
    print(twin)
    print()

    # Add to remote vars
    bv.remote_vars["numbers"] = twin

    print("\nRemote vars registry:")
    print(bv.remote_vars)
    print()

    # Verify it's marked as Twin
    remote_var = bv.remote_vars.get("numbers")
    assert "Twin" in remote_var.var_type

    print("✓ Twin added to remote vars")
    print()


def test_twin_public_only():
    """Test receiving a Twin (public-only)."""
    print("=" * 60)
    print("Test 5: Public-Only Twin (Remote Perspective)")
    print("=" * 60)

    # Simulate receiving a twin from remote
    twin = Twin.public_only(
        public="mock_data.csv",
        owner="alice",
        name="remote_dataset"
    )

    print(twin)
    print()

    assert not twin.has_private
    assert twin.has_public
    assert twin.value == "mock_data.csv"  # Falls back to public

    print("✓ Public-only Twin works")
    print()


if __name__ == "__main__":
    test_twin_basic()
    test_twin_from_mock()
    test_twin_serialization()
    test_twin_remote_vars()
    test_twin_public_only()

    print("=" * 60)
    print("All Twin tests passed! 🎉")
    print("=" * 60)
