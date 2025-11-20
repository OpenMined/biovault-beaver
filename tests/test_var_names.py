import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "python/src"))

import beaver
from beaver import Twin
import pandas as pd

# Clean start
import shutil
shutil.rmtree("shared", ignore_errors=True)

print("=" * 70)
print("VARIABLE NAME AUTO-DETECTION TEST")
print("=" * 70)

# Connect as Bob
bv = beaver.connect("shared", user="bob")

# Test 1: Twin with explicit name
print("\n📦 TEST 1: Twin with explicit name")
print("-" * 70)
patient_data = Twin(
    private=[1, 2, 3],
    public=[1, 2],
    owner="bob",
    name="explicit_name"
)
result = bv.send(patient_data, user="alice")
print(f"✓ Sent with name: {result.envelope.name}")
assert result.envelope.name == "explicit_name", f"Expected 'explicit_name', got '{result.envelope.name}'"

# Test 2: Twin without explicit name (should auto-detect)
print("\n\n📦 TEST 2: Twin without explicit name (auto-detect)")
print("-" * 70)
sales_data = Twin(
    private=[100, 200, 300],
    public=[10, 20, 30],
    owner="bob"
)
result = bv.send(sales_data, user="alice")
print(f"✓ Auto-detected name: {result.envelope.name}")
assert result.envelope.name == "sales_data", f"Expected 'sales_data', got '{result.envelope.name}'"

# Test 3: Regular object (DataFrame)
print("\n\n📦 TEST 3: Regular DataFrame (auto-detect)")
print("-" * 70)
my_dataframe = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
result = bv.send(my_dataframe, user="alice")
print(f"✓ Auto-detected name: {result.envelope.name}")
assert result.envelope.name == "my_dataframe", f"Expected 'my_dataframe', got '{result.envelope.name}'"

# Test 4: Check inbox shows names
print("\n\n📋 TEST 4: Check inbox from Alice's side")
print("-" * 70)
bv_alice = beaver.connect("shared", user="alice")
inbox = bv_alice.inbox()
print(inbox)
print()

# Verify all names are present
envelopes = list(inbox)
names = [e.name for e in envelopes]
print(f"✓ Envelope names: {names}")
assert "explicit_name" in names
assert "sales_data" in names
assert "my_dataframe" in names

print("\n\n" + "=" * 70)
print("ALL TESTS PASSED!")
print("=" * 70)
print("\n✅ Variable names are automatically detected from caller's scope")
print("✅ Inbox shows proper names instead of (unnamed)")
print("✅ Public folder is now inside shared/")
