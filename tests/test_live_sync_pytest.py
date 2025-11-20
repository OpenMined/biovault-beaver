"""Test live synchronization between owner and subscribers."""

import shutil
import time
from pathlib import Path

import pytest

import beaver
from beaver import Twin


@pytest.fixture
def clean_shared():
    """Clean shared directory before and after tests."""
    shared = Path("shared")
    if shared.exists():
        shutil.rmtree(shared)
    yield
    if shared.exists():
        shutil.rmtree(shared)


def test_live_sync_basic(clean_shared):
    """Test basic live sync between owner and subscriber."""
    # Setup
    bv_bob = beaver.connect("shared", user="bob", auto_load_replies=False)
    bv_alice = beaver.connect("shared", user="alice", auto_load_replies=False)

    # Bob creates a Twin with live sync enabled
    counter = Twin(private=1, public=1, owner="bob", name="counter")
    counter.enable_live(interval=0.5)  # Fast interval for testing
    bv_bob.remote_vars.add(counter)

    # Alice loads Bob's Twin (should auto-enable live sync)
    alice_counter = bv_alice.peer("bob").remote_vars["counter"].load(inject=False)

    # Verify Alice's Twin has live sync enabled
    assert alice_counter.live_enabled is True
    assert alice_counter._live_enabled is True
    assert alice_counter.public == 1

    # Bob updates the value
    counter.public = 5

    # Wait for live sync (Bob publishes + Alice reloads)
    time.sleep(1.5)

    # Alice's copy should be updated
    assert alice_counter.public == 5

    # Bob updates again
    counter.public = 10
    time.sleep(1.5)

    # Alice's copy updates again
    assert alice_counter.public == 10

    # Cleanup
    counter.disable_live()
    alice_counter.disable_live()


def test_live_sync_registry_tracking(clean_shared):
    """Test that Alice's Twin tracks registry updates for new file locations."""
    # Setup
    bv_bob = beaver.connect("shared", user="bob", auto_load_replies=False)
    bv_alice = beaver.connect("shared", user="alice", auto_load_replies=False)

    # Bob creates and publishes with live sync
    counter = Twin(private=1, public=1, owner="bob", name="counter")
    counter.enable_live(interval=0.5)
    bv_bob.remote_vars.add(counter)

    # Get initial data location
    registry = bv_bob.remote_vars.registry_path
    import json
    with open(registry) as f:
        data1 = json.load(f)
        location1 = data1["counter"]["data_location"]

    # Alice loads
    alice_counter = bv_alice.peer("bob").remote_vars["counter"].load(inject=False)
    initial_source = alice_counter._source_path

    # Verify initial source path matches registry
    assert initial_source == location1

    # Bob updates (creates new .beaver file)
    counter.public = 5
    time.sleep(1.5)

    # Check registry has new location
    with open(registry) as f:
        data2 = json.load(f)
        location2 = data2["counter"]["data_location"]

    # Location should have changed (new file)
    assert location2 != location1

    # Alice's source path should have updated
    assert alice_counter._source_path == location2

    # And her value should be updated
    assert alice_counter.public == 5

    # Cleanup
    counter.disable_live()
    alice_counter.disable_live()


def test_live_sync_multiple_subscribers(clean_shared):
    """Test live sync with multiple subscribers."""
    # Setup
    bv_bob = beaver.connect("shared", user="bob", auto_load_replies=False)
    bv_alice = beaver.connect("shared", user="alice", auto_load_replies=False)
    bv_charlie = beaver.connect("shared", user="charlie", auto_load_replies=False)

    # Bob creates with live sync
    counter = Twin(private=1, public=1, owner="bob", name="counter")
    counter.enable_live(interval=0.5)
    bv_bob.remote_vars.add(counter)

    # Both Alice and Charlie load
    alice_counter = bv_alice.peer("bob").remote_vars["counter"].load(inject=False)
    charlie_counter = bv_charlie.peer("bob").remote_vars["counter"].load(inject=False)

    # Both should have live sync enabled
    assert alice_counter._live_enabled is True
    assert charlie_counter._live_enabled is True

    # Bob updates
    counter.public = 42
    time.sleep(1.5)

    # Both subscribers should see the update
    assert alice_counter.public == 42
    assert charlie_counter.public == 42

    # Cleanup
    counter.disable_live()
    alice_counter.disable_live()
    charlie_counter.disable_live()


def test_live_sync_display_status(clean_shared):
    """Test that live sync status is correctly displayed."""
    # Setup
    bv_bob = beaver.connect("shared", user="bob", auto_load_replies=False)
    bv_alice = beaver.connect("shared", user="alice", auto_load_replies=False)

    # Bob creates with live sync
    counter = Twin(private=1, public=1, owner="bob", name="counter")
    counter.enable_live(interval=2.0)
    bv_bob.remote_vars.add(counter)

    # Check Bob's display shows live enabled
    display = str(counter)
    assert "Live: 🟢 Enabled" in display
    assert "2.0s" in display

    # Alice loads
    alice_counter = bv_alice.peer("bob").remote_vars["counter"].load(inject=False)

    # Check Alice's display shows live enabled
    alice_display = str(alice_counter)
    assert "Live: 🟢 Enabled" in alice_display
    assert "2.0s" in alice_display

    # Cleanup
    counter.disable_live()
    alice_counter.disable_live()


def test_live_sync_no_private_data(clean_shared):
    """Test that live sync doesn't expose private data to subscribers."""
    # Setup
    bv_bob = beaver.connect("shared", user="bob", auto_load_replies=False)
    bv_alice = beaver.connect("shared", user="alice", auto_load_replies=False)

    # Bob creates Twin with different private/public values
    secret = Twin(private=999, public=1, owner="bob", name="secret")
    secret.enable_live(interval=0.5)
    bv_bob.remote_vars.add(secret)

    # Alice loads
    alice_secret = bv_alice.peer("bob").remote_vars["secret"].load(inject=False)

    # Alice should NOT have private data
    assert alice_secret.private is None
    assert alice_secret.public == 1

    # Bob updates public (but private stays 999)
    secret.public = 5
    time.sleep(1.5)

    # Alice sees public update
    assert alice_secret.public == 5
    # But still no private access
    assert alice_secret.private is None

    # Cleanup
    secret.disable_live()
    alice_secret.disable_live()


def test_live_sync_without_live_enabled(clean_shared):
    """Test that Twins without live sync don't auto-enable on subscriber side."""
    # Setup
    bv_bob = beaver.connect("shared", user="bob", auto_load_replies=False)
    bv_alice = beaver.connect("shared", user="alice", auto_load_replies=False)

    # Bob creates WITHOUT live sync
    counter = Twin(private=1, public=1, owner="bob", name="counter")
    bv_bob.remote_vars.add(counter)

    # Alice loads
    alice_counter = bv_alice.peer("bob").remote_vars["counter"].load(inject=False)

    # Alice's Twin should NOT have live sync enabled
    assert alice_counter.live_enabled is False
    assert getattr(alice_counter, '_live_enabled', False) is False

    # Bob updates
    counter.public = 5
    time.sleep(1.0)

    # Alice's copy should NOT update (no live sync)
    assert alice_counter.public == 1


def test_live_sync_subscriber_mode_detection(clean_shared):
    """Test that subscriber mode is correctly detected based on Twin state."""
    # Setup
    bv_bob = beaver.connect("shared", user="bob", auto_load_replies=False)
    bv_alice = beaver.connect("shared", user="alice", auto_load_replies=False)

    # Bob creates with live sync
    counter = Twin(private=1, public=1, owner="bob", name="counter")
    counter.enable_live(interval=0.5)
    bv_bob.remote_vars.add(counter)

    # Alice loads
    alice_counter = bv_alice.peer("bob").remote_vars["counter"].load(inject=False)

    # Alice's Twin should be in subscriber mode
    # (has _source_path, no private data)
    assert hasattr(alice_counter, '_source_path')
    assert alice_counter._source_path is not None
    assert alice_counter.private is None

    # Bob's Twin should be in owner mode
    # (has private data, is published)
    assert counter.private is not None
    assert hasattr(counter, '_published_registry')

    # Cleanup
    counter.disable_live()
    alice_counter.disable_live()
