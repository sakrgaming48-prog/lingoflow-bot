"""Quick test for the version extractor and import chain."""
from bot import _extract_version

tests = [
    ("models/gemini-2.5-flash", 2.5),
    ("models/gemini-3.1-flash-lite", 3.1),
    ("gemini-1.5-flash-latest", 1.5),
    ("unknown-format", 0.0),
    ("gemini-2.5-flash", 2.5),
]

for name, expected in tests:
    result = _extract_version(name)
    status = "PASS ✅" if result == expected else "FAIL ❌"
    print(f"  {name:40s} → {result}  (expected {expected})  {status}")
    assert result == expected, f"FAILED for {name}"

print("\nAll version extractor tests passed! 🎉")
