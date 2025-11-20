import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "python/src"))

import shutil
import time
import beaver
from beaver import Twin

# Clean start
shutil.rmtree("shared", ignore_errors=True)

print("=" * 70)
print("COMPLETE TWIN WORKFLOW TEST")
print("=" * 70)

# Setup
bv_bob = beaver.connect("shared", user="bob", auto_load_replies=False)
bv_alice = beaver.connect("shared", user="alice", auto_load_replies=False)

# Bob creates Twin
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
    return {
        'count': len(data),
        'sum': sum(data),
        'avg': sum(data) / len(data)
    }

my_result = analyze_data(patient_data_alice)
my_result.request_private()

# Bob loads request
comp_request = list(bv_bob.inbox())[0].load()

print("\n" + "=" * 70)
print("TEST 1: run() - Real data only, returns Twin")
print("=" * 70)

comp_result = comp_request.run()
print(f"\n✓ Result type: {type(comp_result).__name__}")
print(f"✓ Result.data type: {type(comp_result.data).__name__}")
print(f"\n{comp_result}")

assert isinstance(comp_result.data, Twin), "Result should be a Twin"
assert comp_result.data.has_private, "Twin should have private side"
assert not comp_result.data.has_public, "Twin should NOT have public side (run() only)"
assert comp_result.data.private['count'] == 5, "Should execute on real data"
print("\n✅ run() returns Twin with private side only")

print("\n" + "=" * 70)
print("TEST 2: run_both() - Mock + Real, returns Twin with both")
print("=" * 70)

# Reload request for fresh test
comp_request2 = list(bv_bob.inbox())[0].load()
comp_result2 = comp_request2.run_both()

print(f"\n{comp_result2.data}")

assert isinstance(comp_result2.data, Twin), "Result should be a Twin"
assert comp_result2.data.has_private, "Twin should have private side"
assert comp_result2.data.has_public, "Twin should have public side"
assert comp_result2.data.public['count'] == 3, "Public should be mock (count=3)"
assert comp_result2.data.private['count'] == 5, "Private should be real (count=5)"
print("\n✅ run_both() returns Twin with both sides")

print("\n" + "=" * 70)
print("TEST 3: approve() - Send private result")
print("=" * 70)

comp_result2.approve()

# Check Alice's inbox
time.sleep(0.5)
alice_inbox = list(bv_alice.inbox())
if len(alice_inbox) > 0:
    latest = alice_inbox[-1]
    print(f"\n✅ Alice received: {latest.name}")
    received = latest.load()
    print(f"✅ Received data: {received}")
    assert received['count'] == 5, "Should receive real data (count=5)"

print("\n" + "=" * 70)
print("TEST 4: approve_mock() - Send mock result for iteration")
print("=" * 70)

# New request
@bv_alice
def analyze_v2(data):
    return {'v2_count': len(data) * 2}

my_result_v2 = analyze_v2(patient_data_alice)
my_result_v2.request_private()

comp_request3 = list(bv_bob.inbox())[-1].load()
comp_result3 = comp_request3.run_both()

print(f"\nBob's result Twin:")
print(comp_result3.data)

comp_result3.approve_mock()

# Check Alice received mock
time.sleep(0.5)
alice_inbox = list(bv_alice.inbox())
latest = alice_inbox[-1].load()
print(f"\n✅ Alice received mock: {latest}")
assert latest['v2_count'] == 6, "Should receive mock result (3*2=6)"

print("\n" + "=" * 70)
print("TEST 5: reject() - Reject with message")
print("=" * 70)

# New request with fresh function
@bv_alice
def analyze_v3(data):
    return {'v3_sum': sum(data)}

my_result_v3 = analyze_v3(patient_data_alice)
my_result_v3.request_private()

comp_request4 = list(bv_bob.inbox())[-1].load()
comp_result4 = comp_request4.run()

comp_result4.reject(message="Data too sensitive to share")

# Check Alice received rejection
time.sleep(0.5)
alice_inbox = list(bv_alice.inbox())
latest = alice_inbox[-1]
print(f"\n✅ Alice received rejection: {latest.name}")
rejection = latest.load()
print(f"✅ Rejection message: {rejection.get('message')}")
assert rejection.get('_beaver_rejection') == True, "Should be rejection"

print("\n" + "=" * 70)
print("ALL TESTS PASSED!")
print("=" * 70)

print("""
✅ run()         → Twin(private=real, public=None)
✅ run_both()    → Twin(private=real, public=mock)
✅ approve()     → Sends private result
✅ approve_mock()→ Sends mock for iteration
✅ reject()      → Sends rejection message
""")
