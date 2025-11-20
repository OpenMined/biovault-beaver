import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "python/src"))

import shutil
import beaver
from beaver import Twin

# Clean start
shutil.rmtree("shared", ignore_errors=True)

print("=" * 70)
print("TWIN VISUAL STATES - COMPREHENSIVE DEMO")
print("=" * 70)

# Setup
bv_bob = beaver.connect("shared", user="bob", auto_load_replies=False)
bv_alice = beaver.connect("shared", user="alice", auto_load_replies=False)

print("\n" + "=" * 70)
print("SCENARIO 1: Bob creates Twin with BOTH sides (yellow warning)")
print("=" * 70)

patient_data = Twin(
    private=[100, 200, 300, 400, 500],  # Real: 5 patients
    public=[10, 20, 30],                  # Mock: 3 patients
    owner="bob",
    name="patient_data"
)
print(patient_data)

print("\n" + "=" * 70)
print("SCENARIO 2: Bob sends Twin, Alice receives PUBLIC only (green safe)")
print("=" * 70)

bv_bob.send(patient_data, user="alice")
patient_data_alice = list(bv_alice.inbox())[0].load()
print(patient_data_alice)

print("\n" + "=" * 70)
print("SCENARIO 3: Alice creates computation, requests private")
print("=" * 70)

@bv_alice
def analyze_data(data):
    return {
        'count': len(data),
        'sum': sum(data),
        'avg': sum(data) / len(data)
    }

my_result = analyze_data(patient_data_alice)
print("\nAlice's result Twin (pending):")
print(my_result)

my_result.request_private()

print("\n" + "=" * 70)
print("SCENARIO 4: Bob runs computation and gets results")
print("=" * 70)

comp_request = list(bv_bob.inbox())[0].load()

print("\n--- Bob runs on REAL data only (red warning) ---")
real_result = comp_request.run()
print(real_result)

print("\n" + "=" * 70)
print("SCENARIO 5: Bob runs on BOTH mock and real (yellow warning)")
print("=" * 70)

comp_request2 = list(bv_bob.inbox())[0].load()
both_result = comp_request2.run_both()
print(both_result)

print("\n" + "=" * 70)
print("VISUAL SAFETY SUMMARY")
print("=" * 70)
print("""
The color-coded system helps prevent data leakage:

🔒 RED (REAL DATA - SENSITIVE):
   - You have actual sensitive data
   - Be extremely careful
   - Think twice before sharing

🌍 GREEN (MOCK DATA - SAFE):
   - Only synthetic/mock data
   - Safe for development
   - Share freely for testing

⚠️  YELLOW (REAL + MOCK DATA):
   - You have BOTH sides
   - Double-check which you're using
   - .value will use private (red) by default

⏳ PURPLE (PENDING):
   - Waiting for data owner's approval
   - No data available yet
   - Use .request_private() to ask

The visual indicators make it immediately obvious:
- What kind of data you're working with
- Which side .value will return
- When to be extra cautious
""")
