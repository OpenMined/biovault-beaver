import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "python/src"))

import shutil
import beaver
from beaver import Twin

# Clean start
shutil.rmtree("shared", ignore_errors=True)

print("=" * 70)
print("COMPUTATION REQUEST ENVELOPE DISPLAY TEST")
print("=" * 70)

# Setup
bv_bob = beaver.connect("shared", user="bob", auto_load_replies=False)
bv_alice = beaver.connect("shared", user="alice", auto_load_replies=False)

# Bob creates Twin with patient data
patient_data = Twin(
    private=[100, 200, 300, 400, 500],
    public=[10, 20, 30],
    owner="bob",
    name="patient_data"
)
bv_bob.send(patient_data, user="alice")

# Alice receives
patient_data_alice = list(bv_alice.inbox())[0].load()

# Define function with multiple argument types
@bv_alice
def complex_analysis(twin_data, threshold=50, use_avg=True, multiplier=2.0):
    """Function with Twin, static int, bool, and float arguments."""
    count = len(twin_data)
    total = sum(twin_data)
    avg = total / count if count > 0 else 0

    result = {
        'count': count,
        'total': total * multiplier,
        'avg': avg if use_avg else None,
        'above_threshold': len([x for x in twin_data if x > threshold])
    }
    return result

# Call with mixed argument types
my_analysis = complex_analysis(patient_data_alice, threshold=25, use_avg=True, multiplier=1.5)
my_analysis.request_private()

# Bob views the envelope before loading
print("\n" + "=" * 70)
print("Bob's inbox - viewing computation request envelope:")
print("=" * 70)

inbox = list(bv_bob.inbox())
comp_envelope = inbox[0]

print()
print(comp_envelope)

print("\n" + "=" * 70)
print("EXPECTED INFORMATION")
print("=" * 70)
print("""
The envelope should show:

✅ Arguments section:
   - [0] Twin with privacy status (🔒 PRIVATE/🌍 PUBLIC/⚠️ REAL+MOCK)
   - Twin name and type
   - Owner information
   - Data availability indicators

✅ Keyword Arguments section:
   - threshold= int: 25 (with 📌 Static value indicator)
   - use_avg= bool: True (with 📌 Static value indicator)
   - multiplier= float: 1.5 (with 📌 Static value indicator)

✅ Function signature and source code

This gives Bob complete context before loading:
- What data the computation needs
- What static parameters are set
- What the function does
- Whether he has the required data
""")
