"""
================================================================================
EXPERIMENT 05
Title: Porter Stemmer Algorithm for Word Stemming
Objective: Use the Porter Stemmer algorithm from NLTK to perform word stemming
           on a diverse list of words, explain its transformation phases, and
           demonstrate sentence-level stemming.
================================================================================
"""

import nltk
from nltk.stem import PorterStemmer

# Ensure tokenizers are downloaded
try:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
except Exception:
    pass


def explain_porter_phases():
    """Prints an educational overview of Martin Porter's 5-step algorithm."""
    print("\n--- Understanding Martin Porter's 5-Phase Algorithm ---")
    phases = [
        ("Step 1a/1b/1c", "Inflectional endings: Plurals (-sses -> -ss, -ies -> -i, -s -> ''), "
                          "past tense & progressive (-ed, -ing, -y -> -i)."),
        ("Step 2", "Derivational suffixes: -ational -> -ate, -tional -> -tion, -izer -> -ize, -fulness -> -ful."),
        ("Step 3", "Further suffix transformations: -icate -> -ic, -ative -> '', -alize -> -al, -ness -> ''."),
        ("Step 4", "Removal of remaining derivational affixes: -al, -ance, -ence, -er, -ic, -able, -ible, -ant."),
        ("Step 5a/5b", "Cleanup: Removal of silent terminal '-e' and double consonant reduction (e.g., -ll -> -l).")
    ]
    for step, desc in phases:
        print(f"  * {step:<14}: {desc}")
    print()


def main():
    print("=" * 75)
    print("EXPERIMENT 05: PORTER STEMMER ALGORITHM USING NLTK")
    print("=" * 75)

    stemmer = PorterStemmer()
    explain_porter_phases()

    # 1. Stemming a curated list of words grouped by linguistic behavior
    word_categories = {
        "Plurals & Verb Inflections": ["connects", "connecting", "connected", "connection", "connections"],
        "Plural / Third Person Suffixes": ["caresses", "ponies", "ties", "cats", "feed", "agreed"],
        "Adverb & Adjective Derivatives": ["relational", "conditional", "rational", "playfully", "happiness"],
        "Irregular / Edge Cases": ["went", "better", "feet", "mice", "university", "universe"],
        "Morphological Families": ["program", "programming", "programmer", "programs", "programmable"]
    }

    print("[1] Word Stemming Across Morphological Categories:")
    print("-" * 75)
    print(f"{'Category':<32} | {'Original Word':<16} | {'Porter Stem'}")
    print("-" * 75)

    for category, words in word_categories.items():
        for i, word in enumerate(words):
            cat_display = category if i == 0 else ""
            stemmed = stemmer.stem(word)
            print(f"{cat_display:<32} | {word:<16} | {stemmed}")
        print("-" * 75)

    # 2. Sentence-Level Stemming
    sample_sentence = (
        "The programmers were programming complex algorithms while studying "
        "computational linguistics and analyzing textual data systematically."
    )

    print("\n[2] Sentence-Level Stemming Demonstration:")
    print("Original Sentence:\n ", sample_sentence)

    tokens = nltk.word_tokenize(sample_sentence)
    stemmed_tokens = [stemmer.stem(t) for t in tokens]

    print("\nToken-by-Token Comparison:")
    print("-" * 75)
    print(f"{'Original Token':<20} | {'Porter Stem':<20} | {'Changed?'}")
    print("-" * 75)
    for orig, stem in zip(tokens, stemmed_tokens):
        changed = "YES" if orig.lower() != stem else "NO"
        print(f"{orig:<20} | {stem:<20} | {changed}")
    print("-" * 75)

    reconstructed = " ".join(stemmed_tokens)
    print("\nReconstructed Stemmed Text:\n ", reconstructed)


if __name__ == "__main__":
    main()
