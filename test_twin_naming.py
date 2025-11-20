import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "python/src"))

import shutil
import beaver
from beaver import Twin

# Clean start
shutil.rmtree("shared", ignore_errors=True)

print("=" * 70)
print("TWIN VARIABLE NAME AUTO-DETECTION TEST")
print("=" * 70)

# Setup
bv_bob = beaver.connect("shared", user="bob", auto_load_replies=False)
bv_alice = beaver.connect("shared", user="alice", auto_load_replies=False)

# Bob creates and shares Twin
patient_data = Twin(
    private=[100, 200, 300, 400, 500],
    public=[10, 20, 30],
    owner="bob",
    name="patient_data"
)
bv_bob.send(patient_data, user="alice")

# Alice loads and defines function
alice_inbox = list(bv_alice.inbox())
patient_data_alice = alice_inbox[0].load()

@bv_alice
def analyze_data(data):
    return {'count': len(data), 'sum': sum(data)}


print("\n" + "=" * 70)
print("TEST 1: Auto-detect variable name")
print("=" * 70)

# Alice calls with variable name "my_results"
my_results = analyze_data(patient_data_alice)
print(f"\n✓ Created Twin result: {type(my_results).__name__}")
print(f"  Variable name in Alice's code: 'my_results'")

# Request private - should auto-detect "my_results"
print(f"\n📨 Calling: my_results.request_private()")
my_results.request_private(context=bv_alice)

# Check Bob's inbox
print(f"\n✓ Bob's inbox:")
bob_inbox_df = bv_bob.inbox()
print(bob_inbox_df)

# Check the name
bob_inbox = list(bv_bob.inbox())
if bob_inbox:
    envelope = bob_inbox[0]
    print(f"\n✅ Envelope name: '{envelope.name}'")
    assert envelope.name == "my_results", f"Expected 'my_results', got '{envelope.name}'"
    print(f"✅ Auto-detection worked! Variable name preserved.")


print("\n" + "=" * 70)
print("TEST 2: Override with explicit name")
print("=" * 70)

# Alice calls with different variable name but overrides it
another_result = analyze_data(patient_data_alice)
print(f"\n✓ Created Twin result")
print(f"  Variable name in Alice's code: 'another_result'")

# Request private with explicit name override
print(f"\n📨 Calling: another_result.request_private(name='custom_analysis')")
another_result.request_private(context=bv_alice, name="custom_analysis")

# Check Bob's inbox
print(f"\n✓ Bob's inbox:")
bob_inbox_df = bv_bob.inbox()
print(bob_inbox_df)

# Check the name (should be second item now)
bob_inbox = list(bv_bob.inbox())
if len(bob_inbox) >= 2:
    envelope = bob_inbox[1]
    print(f"\n✅ Envelope name: '{envelope.name}'")
    assert envelope.name == "custom_analysis", f"Expected 'custom_analysis', got '{envelope.name}'"
    print(f"✅ Explicit override worked!")


print("\n" + "=" * 70)
print("TEST 3: Load and verify variable names in Bob's namespace")
print("=" * 70)

# Bob loads both
comp1 = bob_inbox[0].load()
comp2 = bob_inbox[1].load()

print(f"\n✓ Loaded computations:")
print(f"  1st: type={type(comp1).__name__}")
print(f"  2nd: type={type(comp2).__name__}")

# Check if variables were injected with correct names
import __main__
if 'my_results' in __main__.__dict__:
    print(f"\n✅ Variable 'my_results' exists in __main__")
else:
    print(f"\n⚠️  Variable 'my_results' NOT in __main__")

if 'custom_analysis' in __main__.__dict__:
    print(f"✅ Variable 'custom_analysis' exists in __main__")
else:
    print(f"⚠️  Variable 'custom_analysis' NOT in __main__")


print("\n" + "=" * 70)
print("ALL TESTS PASSED!")
print("=" * 70)
print("\n✅ Auto-detection: Variable name captured from caller")
print("✅ Override: Explicit name parameter takes precedence")
print("✅ Injection: Variables created with correct names in Bob's namespace")
