"""
================================================================================
EXPERIMENT 16
Title: Named Entity Recognition (NER) using SpaCy
Objective: Implement a Python program using SpaCy to perform Named Entity
           Recognition (NER), extract entity types (PERSON, ORG, GPE, DATE,
           MONEY, etc.), character offsets, and entity explanations.
================================================================================
"""

import sys

# Entity category descriptions according to OntoNotes 5 / SpaCy standard
ENTITY_DESCRIPTIONS = {
    "PERSON": "People, including fictional characters.",
    "ORG": "Companies, agencies, institutions, and organizations.",
    "GPE": "Countries, cities, states, and geopolitical entities.",
    "LOC": "Non-GPE locations, mountain ranges, bodies of water.",
    "DATE": "Absolute or relative dates or periods.",
    "TIME": "Times smaller than a day.",
    "MONEY": "Monetary values, including unit currencies.",
    "PERCENT": "Percentage values (e.g. 25%).",
    "FAC": "Buildings, airports, highways, bridges.",
    "PRODUCT": "Objects, vehicles, foods, etc. (not services).",
    "EVENT": "Named hurricanes, battles, wars, sports events.",
    "WORK_OF_ART": "Titles of books, songs, paintings.",
    "LAW": "Named laws and legal documents.",
    "LANGUAGE": "Any named language.",
    "NORP": "Nationalities or religious or political groups.",
    "CARDINAL": "Numerals that do not fall under another type."
}


def run_spacy_ner(text: str):
    """Executes SpaCy NER using en_core_web_sm if available, or blank model with EntityRuler."""
    try:
        import spacy
    except ImportError:
        print("[!] SpaCy library is not installed in the current environment.")
        print("    To install: pip install spacy && python -m spacy download en_core_web_sm")
        print("\nFalling back to NLTK-based Named Entity Extraction for demonstration...\n")
        run_nltk_fallback_ner(text)
        return

    # Try loading en_core_web_sm
    nlp = None
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("[!] Model 'en_core_web_sm' not found locally. Attempting to initialize pipeline...")
        try:
            # Fallback to spacy blank English with EntityRuler for offline demo
            nlp = spacy.blank("en")
            ruler = nlp.add_pipe("entity_ruler")
            patterns = [
                {"label": "ORG", "pattern": "Google Antigravity"},
                {"label": "ORG", "pattern": "OpenAI"},
                {"label": "ORG", "pattern": "Apple"},
                {"label": "PERSON", "pattern": "Sundar Pichai"},
                {"label": "PERSON", "pattern": "Alan Turing"},
                {"label": "GPE", "pattern": "California"},
                {"label": "GPE", "pattern": "London"},
                {"label": "GPE", "pattern": "India"},
                {"label": "DATE", "pattern": "2026"},
                {"label": "DATE", "pattern": "yesterday"},
                {"label": "MONEY", "pattern": "$100 million"}
            ]
            ruler.add_patterns(patterns)
            print("    Configured SpaCy EntityRuler with patterns.")
        except Exception as e:
            print(f"    Failed initializing SpaCy pipeline: {e}")
            run_nltk_fallback_ner(text)
            return

    doc = nlp(text)

    print("=" * 80)
    print("SPACY NAMED ENTITY RECOGNITION (NER) RESULTS")
    print("=" * 80)
    print(f"{'Entity Text':<24} | {'Label':<12} | {'Span (Start, End)':<20} | {'Description'}")
    print("-" * 80)

    for ent in doc.ents:
        desc = ENTITY_DESCRIPTIONS.get(ent.label_, "Named entity")
        span_str = f"chars ({ent.start_char}, {ent.end_char})"
        print(f"{ent.text:<24} | {ent.label_:<12} | {span_str:<20} | {desc}")
    print("-" * 80)

    print("\nVisual Entity Bracket Display:")
    rendered = text
    # Replace entities with annotated tags in reverse order of position
    for ent in sorted(doc.ents, key=lambda e: e.start_char, reverse=True):
        rendered = rendered[:ent.start_char] + f"[{ent.text}]({ent.label_})" + rendered[ent.end_char:]
    print(rendered)


def run_nltk_fallback_ner(text: str):
    """Fallback demonstration using NLTK ne_chunk."""
    import nltk
    try:
        nltk.download("maxent_ne_chunker", quiet=True)
        nltk.download("maxent_ne_chunker_tab", quiet=True)
        nltk.download("words", quiet=True)
    except Exception:
        pass

    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    tree = nltk.ne_chunk(tagged)

    print("-" * 80)
    print(f"{'Entity Text':<24} | {'Label':<12} | {'Type'}")
    print("-" * 80)
    for subtree in tree:
        if hasattr(subtree, "label"):
            entity_name = " ".join(c[0] for c in subtree.leaves())
            entity_type = subtree.label()
            print(f"{entity_name:<24} | {entity_type:<12} | Named Entity")
    print("-" * 80)


def main():
    print("=" * 80)
    print("EXPERIMENT 16: NAMED ENTITY RECOGNITION (NER) USING SPACY")
    print("=" * 80)

    sample_text = (
        "Sundar Pichai announced that Google Antigravity invested $100 million in California in 2026. "
        "Meanwhile, OpenAI and Apple are collaborating in London to build next-generation language systems."
    )

    print("Input Text:\n", sample_text)
    print()
    run_spacy_ner(sample_text)


if __name__ == "__main__":
    main()
