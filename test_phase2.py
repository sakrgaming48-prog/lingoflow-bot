"""Quick smoke tests for the Gemini JSON parser."""

import json
from bot import _parse_gemini_json

# ── Test data ────────────────────────────────────────────────
VALID_ITEM = json.dumps([{
    "term": "scrutinize",
    "arabic": "يفحص بدقة",
    "definition": "To examine something very carefully.",
    "example": "The doctor scrutinized the lab results.",
    "synonym": "examine",
    "source_context": "general",
}])

# Test 1: Clean JSON
r = _parse_gemini_json(VALID_ITEM)
assert len(r) == 1 and r[0]["term"] == "scrutinize"
print("Test 1 (clean JSON):            PASS ✅")

# Test 2: Code-fenced JSON
r = _parse_gemini_json(f"```json\n{VALID_ITEM}\n```")
assert len(r) == 1
print("Test 2 (code-fenced):           PASS ✅")

# Test 3: Missing keys → skipped
r = _parse_gemini_json('[{"term": "hello"}]')
assert len(r) == 0
print("Test 3 (missing keys skipped):  PASS ✅")

# Test 4: Empty array
r = _parse_gemini_json("[]")
assert len(r) == 0
print("Test 4 (empty array):           PASS ✅")

# Test 5: Invalid source_context defaults to general
bad_ctx = VALID_ITEM.replace("general", "unknown")
r = _parse_gemini_json(bad_ctx)
assert r[0]["source_context"] == "general"
print("Test 5 (bad context → general): PASS ✅")

# Test 6: Malformed JSON raises JSONDecodeError
try:
    _parse_gemini_json("not json at all")
    assert False, "Should have raised"
except json.JSONDecodeError:
    print("Test 6 (malformed → exception): PASS ✅")

# Test 7: Multiple items, one invalid
multi = json.dumps([
    {
        "term": "good",
        "arabic": "جيد",
        "definition": "def",
        "example": "ex",
        "synonym": "syn",
        "source_context": "general",
    },
    {"term": "bad_incomplete"},  # missing keys
    {
        "term": "also_good",
        "arabic": "أيضاً جيد",
        "definition": "def2",
        "example": "ex2",
        "synonym": "syn2",
        "source_context": "medical",
    },
])
r = _parse_gemini_json(multi)
assert len(r) == 2 and r[0]["term"] == "good" and r[1]["term"] == "also_good"
print("Test 7 (mixed valid/invalid):   PASS ✅")

# Test 8: DB add_to_cart + duplicate detection
import asyncio
from db import init_db, add_to_cart

async def test_cart():
    db = "_test_parser.db"
    await init_db(db)
    ok1 = await add_to_cart(db, 999, "test", "ت", "d", "e", "s", "general")
    ok2 = await add_to_cart(db, 999, "test", "ت", "d", "e", "s", "general")
    assert ok1 is True,  "First insert should succeed"
    assert ok2 is False, "Duplicate should be skipped"
    import os; os.remove(db)
    print("Test 8 (cart dedup):            PASS ✅")

asyncio.run(test_cart())

print("\n🎉 All tests passed!")
