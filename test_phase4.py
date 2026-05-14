"""Test Phase 4: Anki generation and Media Group cache setup"""
import os
import asyncio
from anki_utils import generate_apkg

async def test_genanki():
    items = [
        {
            "term": "scrutinize",
            "arabic": "يفحص بدقة",
            "definition": "To examine something very carefully.",
            "example": "The doctor scrutinized the lab results before making a diagnosis.",
            "synonym": "examine",
            "source_context": "general"
        },
        {
            "term": "suture",
            "arabic": "خياطة الجروح",
            "definition": "A stitch or row of stitches holding together the edges of a wound.",
            "example": "The surgeon applied a running suture.",
            "synonym": "stitch",
            "source_context": "medical"
        }
    ]
    
    output_path = "test_deck.apkg"
    if os.path.exists(output_path):
        os.remove(output_path)
        
    generate_apkg(12345, "MainDeck", "TestSub", items, output_path)
    
    assert os.path.exists(output_path), "File was not created"
    size = os.path.getsize(output_path)
    assert size > 0, "File is empty"
    
    os.remove(output_path)
    print("Test 1 (genanki output): PASS ✅")

async def main():
    await test_genanki()
    print("All Phase 4 tests passed! 🎉")

if __name__ == "__main__":
    asyncio.run(main())
