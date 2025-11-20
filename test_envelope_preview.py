import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "python/src"))

import shutil
import beaver
from beaver import Twin

# Clean start
shutil.rmtree("shared", ignore_errors=True)

print("=" * 70)
print("ENVELOPE PREVIEW TEST - ANSI Color Stripping")
print("=" * 70)

# Setup
bv_bob = beaver.connect("shared", user="bob", auto_load_replies=False)
bv_alice = beaver.connect("shared", user="alice", auto_load_replies=False)

print("\n" + "=" * 70)
print("TEST 1: Send Twin with BOTH sides (has color codes)")
print("=" * 70)

patient_data = Twin(
    private=[100, 200, 300, 400, 500],
    public=[10, 20, 30],
    owner="bob",
    name="patient_data"
)

print("\nDirect Twin display (with colors):")
print(patient_data)

print("\nSending to Alice...")
bv_bob.send(patient_data, user="alice")

print("\n" + "=" * 70)
print("TEST 2: Check envelope preview (should have NO color codes)")
print("=" * 70)

# List Alice's inbox to see envelope
inbox = list(bv_alice.inbox())
envelope = inbox[0]

print("\nEnvelope display:")
print(envelope)

# Verify no ANSI codes in the string representation
envelope_str = str(envelope)
has_ansi = '\033[' in envelope_str

print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)
if has_ansi:
    print("❌ FAILED: ANSI color codes found in envelope preview!")
    print("   This will display as garbled text.")
else:
    print("✅ SUCCESS: Envelope preview is clean (no ANSI codes)")
    print("   Preview displays correctly without color artifacts.")

print("\n" + "=" * 70)
print("TEST 3: Load Twin and verify it still has colors")
print("=" * 70)

loaded_twin = envelope.load()
loaded_str = str(loaded_twin)
has_color = '\033[' in loaded_str

if has_color:
    print("✅ SUCCESS: Loaded Twin still has color codes")
    print("   Colors work correctly when displaying the actual object.")
else:
    print("⚠️  WARNING: Loaded Twin missing color codes")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
The fix ensures:
1. Envelope previews are clean (no ANSI codes) → readable metadata
2. Loaded objects retain colors → rich interactive display
3. Best of both worlds: clean metadata + colorful objects
""")
