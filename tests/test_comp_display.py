import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "python/src"))

import shutil
import beaver
from beaver import Twin

# Clean start
shutil.rmtree("shared", ignore_errors=True)

print("=" * 70)
print("COMPUTATION REQUEST DISPLAY TEST")
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
def complex_analysis(twin_data, threshold=50, use_avg=True):
    """Function with Twin, static int, and static bool arguments."""
    count = len(twin_data)
    total = sum(twin_data)
    avg = total / count if count > 0 else 0

    result = {
        'count': count,
        'total': total,
        'avg': avg if use_avg else None,
        'above_threshold': len([x for x in twin_data if x > threshold])
    }
    return result

# Call with mixed argument types
my_analysis = complex_analysis(patient_data_alice, threshold=25, use_avg=True)
my_analysis.request_private()

# Bob loads and views
print("\n" + "=" * 70)
print("Bob receives computation request with detailed arg info:")
print("=" * 70)

comp_request = list(bv_bob.inbox())[0].load()
print(comp_request)

print("\n" + "=" * 70)
print("DISPLAY BREAKDOWN")
print("=" * 70)

print("""
✅ Arg [0]: Shows Twin with:
   - Privacy status (🌍 PUBLIC / 🔒 PRIVATE)
   - Twin name and type
   - Owner
   - Data availability indicators

✅ Kwargs: Shows static bound values with:
   - Type information
   - Actual value
   - "Static value" indicator

This gives Bob complete context about:
1. Which arguments are Twins (privacy-sensitive)
2. Which are static literals (no privacy concerns)
3. What data is available for testing (mock vs real)
4. Who owns the data
""")
