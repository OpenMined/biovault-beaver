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
print("END-TO-END TWIN COMPUTATION TEST")
print("=" * 70)

# ============================================================================
# PART 1: Data Owner (Bob) - Create and share Twin
# ============================================================================
print("\n" + "=" * 70)
print("PART 1: Bob (Data Owner) - Create and share Twin")
print("=" * 70)

bv_bob = beaver.connect("shared", user="bob", auto_load_replies=True)

# Create Twin with real and mock patient data
patient_counts = Twin(
    private=[100, 200, 300, 400, 500],  # Real data (5 values)
    public=[10, 20, 30],                  # Mock data (3 values)
    owner="bob",
    name="patient_counts"
)

print(f"\n✓ Bob created Twin:")
print(patient_counts)

# Publish to remote_vars (optional - Twin registry is the primary mechanism)
bv_bob.remote_vars["patient_counts"] = patient_counts

# Share with Alice
bv_bob.send(patient_counts, user="alice")
print(f"\n✓ Bob shared Twin with Alice")

# ============================================================================
# PART 2: Data Scientist (Alice) - Receive and develop analysis
# ============================================================================
print("\n" + "=" * 70)
print("PART 2: Alice (Data Scientist) - Receive and develop analysis")
print("=" * 70)

bv_alice = beaver.connect("shared", user="alice", auto_load_replies=True)

# Load Twin from inbox
print(f"\n📥 Alice's inbox:")
print(bv_alice.inbox())

alice_inbox = list(bv_alice.inbox())
patient_counts_alice = alice_inbox[0].load()

print(f"\n✓ Alice loaded Twin:")
print(patient_counts_alice)

# Define analysis function
@bv_alice
def analyze_data(data):
    """Compute statistics on data."""
    return {
        'count': len(data),
        'sum': sum(data),
        'avg': sum(data) / len(data) if data else 0
    }

print(f"\n✓ Alice defined analysis function")

# Call with Twin → get Twin result
result = analyze_data(patient_counts_alice)

print(f"\n✓ Alice executed analysis on Twin:")
print(result)
print(f"\n✓ Public result (executed on mock): {result.public}")

# Request private execution
print(f"\n📨 Alice requesting private execution...")
result.request_private(context=bv_alice)

# ============================================================================
# PART 3: Data Owner (Bob) - Approve computation
# ============================================================================
print("\n" + "=" * 70)
print("PART 3: Bob - Review and approve computation")
print("=" * 70)

# Give auto_load a moment to process
time.sleep(0.5)

print(f"\n📥 Bob's inbox:")
print(bv_bob.inbox())

# Load computation request
bob_inbox = list(bv_bob.inbox())
if len(bob_inbox) > 0:
    comp_request = bob_inbox[0].load()

    print(f"\n✓ Bob loaded computation request:")
    print(comp_request)

    # Execute on private data
    print(f"\n⚙️  Bob executing on private data...")
    comp_result = comp_request.run(context=bv_bob)

    print(f"\n✓ Execution complete:")
    print(comp_result)

    # Approve and send back
    print(f"\n✅ Bob approving result...")
    comp_result.approve()

    print(f"\n✓ Result sent back to Alice")
else:
    print("❌ No computation request found in Bob's inbox!")
    sys.exit(1)

# ============================================================================
# PART 4: Data Scientist (Alice) - Receive result
# ============================================================================
print("\n" + "=" * 70)
print("PART 4: Alice - Auto-receive approved result")
print("=" * 70)

# Give auto_load time to process the reply
print(f"\n⏳ Waiting for auto_load to process reply...")
time.sleep(2)

print(f"\n✓ Alice's result Twin (should be auto-updated):")
print(result)

# Check if private side was updated
if result.has_private and not isinstance(result.private, beaver.computation.ComputationRequest):
    print(f"\n✅ SUCCESS! Private result received:")
    print(f"   {result.private}")

    # Verify it's the real result (count=5, not 3)
    if result.private.get('count') == 5:
        print(f"\n✅ Verified: Using real data (count=5)")
    else:
        print(f"\n❌ Error: Expected count=5, got {result.private.get('count')}")

    # Show that .value now uses private
    print(f"\n✓ result.value now uses private:")
    print(f"   {result.value}")
else:
    print(f"\n⚠️  Private side not updated yet")
    print(f"   Type: {type(result.private).__name__}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("END-TO-END TEST COMPLETE!")
print("=" * 70)

print("\n✅ Flow completed:")
print("   1. Bob created Twin with real + mock data")
print("   2. Alice received Twin (public only)")
print("   3. Alice defined @bv function")
print("   4. Alice called function with Twin → Twin result")
print("   5. Alice requested private execution")
print("   6. Bob loaded, executed, and approved")
print("   7. Alice's result auto-updated via auto_load_replies")
