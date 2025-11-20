import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "python/src"))

import time
from beaver import Twin

print("=" * 70)
print("LIVE SYNC PRINT SPAM TEST")
print("=" * 70)

# Create a Twin with live sync enabled
patient_data = Twin(
    private=[1, 2, 3, 4, 5],
    public=[1, 2, 3],
    owner="bob",
    name="patient_data"
)

print("\n📦 Created Twin:")
print(patient_data)

print("\n\n🟢 Enabling live sync (2 second interval)...")
patient_data.enable_live(mutable=False, interval=2.0)

print("\n⏱️  Waiting 10 seconds to see if prints spam...")
print("(If working correctly, you should NOT see repeated 'Using PRIVATE data' messages)")
print()

# Wait and see if it spams
for i in range(10):
    time.sleep(1)
    print(f"  {i+1} second(s) elapsed...")

print("\n\n✅ If you didn't see repeated 'Using PRIVATE data' messages above,")
print("   the fix is working correctly!")

print("\n⚫ Disabling live sync...")
patient_data.disable_live()

print("\n✓ Test complete!")
