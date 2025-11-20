import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "python/src"))

import shutil
import beaver
from beaver import Twin

# Clean start
shutil.rmtree("shared", ignore_errors=True)

print("=" * 70)
print("TEST: run_mock() vs run()")
print("=" * 70)

# Setup
bv_bob = beaver.connect("shared", user="bob", auto_load_replies=False)
bv_alice = beaver.connect("shared", user="alice", auto_load_replies=False)

# Bob creates Twin with real + mock data
patient_data = Twin(
    private=[100, 200, 300, 400, 500],  # Real: 5 patients
    public=[10, 20, 30],                  # Mock: 3 patients
    owner="bob",
    name="patient_data"
)
bv_bob.send(patient_data, user="alice")

# Alice receives and creates analysis
patient_data_alice = list(bv_alice.inbox())[0].load()

@bv_alice
def analyze_data(data):
    """Compute statistics."""
    return {
        'count': len(data),
        'sum': sum(data),
        'avg': sum(data) / len(data)
    }

# Alice calls with Twin and requests private
my_result = analyze_data(patient_data_alice)
my_result.request_private()

# Bob loads the request
comp_request = list(bv_bob.inbox())[0].load()

print("\n" + "=" * 70)
print("Bob received computation request:")
print("=" * 70)
print(comp_request)

print("\n" + "=" * 70)
print("TEST 1: run_mock() - Execute on mock/public data")
print("=" * 70)

mock_result = comp_request.run_mock()
print(f"\nMock result: {mock_result}")
assert mock_result['count'] == 3, f"Expected count=3, got {mock_result['count']}"
assert mock_result['sum'] == 60, f"Expected sum=60, got {mock_result['sum']}"
print("✅ Mock execution correct (count=3, sum=60)")

print("\n" + "=" * 70)
print("TEST 2: run() - Execute on real/private data")
print("=" * 70)

comp_result = comp_request.run()
print(f"\nReal result: {comp_result.data}")
assert comp_result.data['count'] == 5, f"Expected count=5, got {comp_result.data['count']}"
assert comp_result.data['sum'] == 1500, f"Expected sum=1500, got {comp_result.data['sum']}"
print("✅ Real execution correct (count=5, sum=1500)")

print("\n" + "=" * 70)
print("WORKFLOW DEMO")
print("=" * 70)

print("""
# Bob's workflow:

1. Load request:
   comp_request = bv.inbox()[0].load()

2. Preview on mock data (safe):
   mock_result = comp_request.run_mock()
   # → {'count': 3, 'sum': 60, 'avg': 20.0}

3. Execute on real data:
   comp_result = comp_request.run()
   # → {'count': 5, 'sum': 1500, 'avg': 300.0}

4. Approve and send back:
   comp_result.approve()
""")

print("\n" + "=" * 70)
print("ALL TESTS PASSED!")
print("=" * 70)
