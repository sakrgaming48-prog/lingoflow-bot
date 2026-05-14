"""
LingoFlow — Anki Generation Utilities
=====================================
Handles the creation of .apkg files using genanki with
deterministic GUIDs to prevent duplicates.
"""

import hashlib
import genanki

CSS = """
.card {
    font-family: 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 20px;
    text-align: center;
    color: #2c3e50;
    background-color: #fcfcfc;
    padding: 15px;
}
.term {
    font-size: 30px;
    font-weight: 700;
    color: #2b2b2b;
    margin-bottom: 18px;
}
.arabic {
    font-size: 26px;
    color: #34495e;
    font-weight: 600;
    margin-bottom: 12px;
    direction: rtl;
    text-align: center;
}
.definition {
    font-size: 18px;
    color: #555555;
    font-style: italic;
    margin-bottom: 15px;
    line-height: 1.4;
}
.example {
    font-size: 17px;
    color: #34495e;
    background-color: #f0f4f8;
    padding: 12px 15px;
    border-radius: 8px;
    margin-bottom: 15px;
    text-align: left;
    border-left: 4px solid #9fb9d1;
    line-height: 1.4;
}
.synonym {
    font-size: 15px;
    color: #7f8c8d;
}

/* Dark Mode (.nightMode) - Soft pastel tones, dark soft grays */
.nightMode .card {
    background-color: #1e2022;
    color: #d1d5db;
}
.nightMode .term { 
    color: #e5e7eb; 
}
.nightMode .arabic { 
    color: #93c5fd; /* Soft pastel blue */
}
.nightMode .definition { 
    color: #a1a1aa; 
}
.nightMode .example { 
    color: #cbd5e1; 
    background-color: #2d3136; 
    border-left: 4px solid #4b5563; 
}
.nightMode .synonym { 
    color: #6b7280; 
}
"""

FRONT_HTML = """
<div class="term">{{Term}}</div>
"""

BACK_HTML = """
{{FrontSide}}
<hr id="answer">
<div class="arabic">{{Arabic}}</div>
<div class="definition">{{Definition}}</div>
<div class="example"><i>"{{Example}}"</i></div>
<div class="synonym">Synonym: {{Synonym}}</div>
"""

def _get_id(seed_str: str) -> int:
    """Generate a deterministic positive integer ID from a string.
    Bound to 2^63 to fit within genanki's limits.
    """
    return int(hashlib.sha256(seed_str.encode('utf-8')).hexdigest(), 16) % (2**63)


def get_model(user_id: int) -> genanki.Model:
    """Returns the LingoFlow Anki Model with deterministic ID."""
    model_id = _get_id(f"{user_id}_lingoflow_model_v1")
    return genanki.Model(
        model_id,
        'LingoFlow Medical & General',
        fields=[
            {'name': 'Term'},
            {'name': 'Arabic'},
            {'name': 'Definition'},
            {'name': 'Example'},
            {'name': 'Synonym'},
            {'name': 'SourceContext'}
        ],
        templates=[
            {
                'name': 'Card 1',
                'qfmt': FRONT_HTML,
                'afmt': BACK_HTML,
            },
        ],
        css=CSS
    )


def generate_apkg(
    user_id: int, 
    main_deck: str, 
    sub_deck: str, 
    cart_items: list[dict], 
    output_path: str
) -> None:
    """Compiles the cart items into an Anki .apkg file."""
    
    full_deck_name = f"{main_deck}::{sub_deck}"
    deck_id = _get_id(f"{user_id}_{full_deck_name}")
    
    deck = genanki.Deck(
        deck_id,
        full_deck_name
    )
    
    model = get_model(user_id)
    
    for item in cart_items:
        note_id = _get_id(f"{user_id}_{item['term']}")
        
        note = genanki.Note(
            model=model,
            fields=[
                item['term'],
                item['arabic'],
                item['definition'],
                item['example'],
                item['synonym'],
                item['source_context']
            ]
        )
        note.guid = str(note_id) # Using the numeric ID as string for GUID
        
        deck.add_note(note)
        
    package = genanki.Package(deck)
    package.write_to_file(output_path)
