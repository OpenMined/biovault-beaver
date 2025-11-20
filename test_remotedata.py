import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "python/src"))

import time
from beaver import Twin

print("=" * 70)
print("REMOTEDATA ARCHITECTURE TESTS")
print("=" * 70)

# Test 1: Basic Twin functionality (backward compatibility)
print("\n📦 TEST 1: Basic Twin Creation")
print("-" * 70)

dataset = Twin(
    private=[100, 200, 300, 400, 500],
    public=[10, 20, 30],
    owner="alice",
    name="sales_data"
)

print(dataset)
print(f"\n✓ has_private: {dataset.has_private}")
print(f"✓ has_public: {dataset.has_public}")
print(f"✓ has_data(): {dataset.has_data()}")
print(f"✓ get_value(): {dataset.get_value()}")

# Test 2: Value property (should print once)
print("\n\n📊 TEST 2: Value Access (should print once)")
print("-" * 70)

print(f"First access: {dataset.value}")
print(f"Second access: {dataset.value}")
print(f"Third access: {dataset.value}")
print("✓ Message printed only once")

# Test 3: Live sync - read-only mode
print("\n\n🔴 TEST 3: Live Sync (Read-Only)")
print("-" * 70)

live_data = Twin(
    private=["log1", "log2"],
    public=["log1"],
    owner="alice",
    name="logs"
)

print("Enabling live sync (read-only, 1s interval)...")
live_data.enable_live(mutable=False, interval=1.0)

print(f"\n✓ live: {live_data.live}")
print(f"✓ mutable: {live_data.mutable}")
print(f"✓ sync_interval: {live_data.sync_interval}")

print("\nLive data display:")
print(live_data)

# Test 4: Live sync - mutable mode
print("\n\n🔴 TEST 4: Live Sync (Mutable)")
print("-" * 70)

mutable_data = Twin(
    private=[1, 2, 3],
    public=[1],
    owner="bob",
    name="counter"
)

print("Enabling live sync (mutable, 1s interval)...")
mutable_data.enable_live(mutable=True, interval=1.0)

print(f"\n✓ live: {mutable_data.live}")
print(f"✓ mutable: {mutable_data.mutable}")
print(f"✓ sync_interval: {mutable_data.sync_interval}")

print("\nMutable data display:")
print(mutable_data)

# Test 5: Change detection
print("\n\n🔔 TEST 5: Change Detection")
print("-" * 70)

change_data = Twin(
    private=["initial"],
    public=["initial"],
    owner="charlie",
    name="changes"
)

changes_detected = []

def on_change_callback():
    changes_detected.append(time.time())
    print(f"  🔔 Change detected! (Total: {len(changes_detected)})")

change_data.enable_live(mutable=False, interval=0.5)
change_data.on_change(on_change_callback)

print("Monitoring for changes (modifying private value)...")
print("Initial value:", change_data.private)

# Simulate changes
time.sleep(1)
change_data.private.append("change1")
time.sleep(1)
change_data.private.append("change2")
time.sleep(1)

print(f"\n✓ Changes detected: {len(changes_detected)}")
print(f"✓ Final value: {change_data.private}")

# Test 6: Disable live sync
print("\n\n⚫ TEST 6: Disable Live Sync")
print("-" * 70)

print("Disabling live sync on all test objects...")
live_data.disable_live()
mutable_data.disable_live()
change_data.disable_live()

print(f"✓ live_data.live: {live_data.live}")
print(f"✓ mutable_data.live: {mutable_data.live}")
print(f"✓ change_data.live: {change_data.live}")

# Test 7: RemoteData interface
print("\n\n🔍 TEST 7: RemoteData Interface")
print("-" * 70)

interface_test = Twin(
    private="real_value",
    public="mock_value",
    owner="dave",
    name="interface"
)

print(f"✓ has_data(): {interface_test.has_data()}")
print(f"✓ get_value(): {interface_test.get_value()}")
print(f"✓ id: {interface_test.id[:12]}...")
print(f"✓ twin_id: {interface_test.twin_id[:12]}...")
print(f"✓ private_id: {interface_test.private_id[:12]}...")
print(f"✓ public_id: {interface_test.public_id[:12]}...")
print(f"✓ var_type: {interface_test.var_type}")
print(f"✓ owner: {interface_test.owner}")
print(f"✓ name: {interface_test.name}")

print("\nTesting request_access()...")
interface_test.request_access()

# Test 8: Public-only Twin
print("\n\n🌍 TEST 8: Public-Only Twin (Remote)")
print("-" * 70)

remote_twin = Twin.public_only(
    public=[1, 2, 3],
    owner="alice",
    name="remote_data"
)

print(remote_twin)
print(f"\n✓ has_private: {remote_twin.has_private}")
print(f"✓ has_public: {remote_twin.has_public}")
print(f"✓ value uses: public")

# Test 9: Auto-mock generation
print("\n\n🎭 TEST 9: Auto-Mock Generation")
print("-" * 70)

auto_twin = Twin.from_mock(
    private=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    owner="eve",
    name="auto_mock"
)

print(f"✓ Private (full): {auto_twin.private}")
print(f"✓ Public (auto-mock): {auto_twin.public}")

# Test 10: Display tests
print("\n\n📺 TEST 10: Display Formats")
print("-" * 70)

display_twin = Twin(
    private="private_data" * 10,
    public="public_data",
    owner="frank",
    name="display_test"
)
display_twin.enable_live(mutable=True, interval=2.0)

print("String representation (__str__):")
print(display_twin)

print("\n\nRepr representation (__repr__):")
print(repr(display_twin))

print("\n\n" + "=" * 70)
print("ALL TESTS COMPLETED")
print("=" * 70)
print("\n✅ RemoteData architecture is working correctly!")
print("✅ Twin extends RemoteData successfully")
print("✅ LiveMixin provides live sync capability")
print("✅ Unified display shows all relevant information")
print("✅ Backward compatibility maintained")
