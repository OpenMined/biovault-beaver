import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "python/src"))

from beaver import Twin

print("=" * 70)
print("TWIN VISUAL STATES TEST")
print("=" * 70)

print("\n" + "=" * 70)
print("STATE 1: PRIVATE ONLY (RED - SENSITIVE)")
print("=" * 70)
private_only = Twin(
    private={"patients": 100, "avg_age": 45.2, "conditions": ["diabetes", "hypertension"]},
    public=None,
    owner="bob",
    name="patient_records"
)
print(private_only)

print("\n" + "=" * 70)
print("STATE 2: PUBLIC ONLY (GREEN - SAFE FOR DEVELOPMENT)")
print("=" * 70)
public_only = Twin(
    private=None,
    public={"patients": 10, "avg_age": 40.0, "conditions": ["example"]},
    owner="alice",
    name="mock_data"
)
print(public_only)

print("\n" + "=" * 70)
print("STATE 3: BOTH SIDES (YELLOW - BE CAREFUL)")
print("=" * 70)
both_sides = Twin(
    private={"real": "sensitive data", "count": 5000},
    public={"real": "mock data", "count": 50},
    owner="bob",
    name="test_results"
)
print(both_sides)

print("\n" + "=" * 70)
print("VISUAL SUMMARY")
print("=" * 70)
print("""
Color Scheme Purpose:
  🔒 RED    = Real sensitive data present - BE CAREFUL!
  🌍 GREEN  = Only mock data - SAFE for development
  ⚠️  YELLOW = Both real + mock - DOUBLE CHECK which you're using
  ⏳ PURPLE = Pending/no data - Waiting for approval

This helps users immediately recognize what kind of data they're working with
and avoid accidentally exposing sensitive information.
""")
