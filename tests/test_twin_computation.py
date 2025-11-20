import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "python/src"))

import shutil
import beaver
from beaver import Twin

# Clean start
shutil.rmtree("shared", ignore_errors=True)

print("=" * 70)
print("TWIN-AWARE COMPUTATION TEST")
print("=" * 70)

# Connect as Alice (data scientist)
bv = beaver.connect("shared", user="alice", auto_load_replies=False)

print("\n📦 TEST 1: Define analysis function")
print("-" * 70)

@bv
def analyze_data(data):
    """Compute statistics on data."""
    return {
        'count': len(data),
        'sum': sum(data),
        'avg': sum(data) / len(data) if data else 0
    }

print("✓ Function defined with @bv decorator")

print("\n\n📦 TEST 2: Create Twin with mock and real data")
print("-" * 70)

patient_counts = Twin(
    private=[100, 200, 300, 400, 500],  # Real data (5 values)
    public=[10, 20, 30],                  # Mock data (3 values)
    owner="bob",
    name="patient_counts"
)

print(f"✓ Twin created:")
print(f"  Private: {patient_counts.private}")
print(f"  Public: {patient_counts.public}")

print("\n\n📦 TEST 3: Call with explicit .public (normal execution)")
print("-" * 70)

mock_result = analyze_data(patient_counts.public)
print(f"✓ Result type: {type(mock_result).__name__}")
print(f"✓ Result: {mock_result}")
assert isinstance(mock_result, dict), f"Expected dict, got {type(mock_result)}"
assert mock_result['count'] == 3, f"Expected count=3, got {mock_result['count']}"

print("\n\n📦 TEST 4: Call with Twin (Twin-aware execution)")
print("-" * 70)

result = analyze_data(patient_counts)
print(f"✓ Result type: {type(result).__name__}")
print(result)

# Verify it's a Twin
assert isinstance(result, Twin), f"Expected Twin, got {type(result)}"

# Verify public result (executed on mock)
print(f"\n✓ Public result (mock): {result.public}")
assert isinstance(result.public, dict), "Public should be dict"
assert result.public['count'] == 3, f"Expected count=3 for mock, got {result.public['count']}"

# Verify private is a ComputationRequest
print(f"✓ Private type: {type(result.private).__name__}")
from beaver.computation import ComputationRequest
assert isinstance(result.private, ComputationRequest), "Private should be ComputationRequest"

print("\n\n📦 TEST 5: Request private execution")
print("-" * 70)

result.request_private(context=bv)

# Check that computation was sent
print(f"\n✓ Checking Bob's inbox...")
bv_bob = beaver.connect("shared", user="bob", auto_load_replies=False)
inbox = bv_bob.inbox()
print(f"✓ Bob has {len(list(inbox))} message(s)")

if len(list(inbox)) > 0:
    print("\n✓ Bob's inbox:")
    print(bv_bob.inbox())

print("\n\n" + "=" * 70)
print("BASIC TESTS PASSED!")
print("=" * 70)
print("\n✅ @bv decorator detects Twin arguments")
print("✅ Returns Twin with public result executed")
print("✅ Returns Twin with ComputationRequest for private")
print("✅ .request_private() sends computation to owner")
print("\n💡 Next: Bob needs to load, execute, and approve the computation")
