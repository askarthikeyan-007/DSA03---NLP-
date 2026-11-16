"""
================================================================================
EXPERIMENT 17
Title: WordNet Lexical Database Exploration
Objective: Demonstrate how to access WordNet in Python using NLTK to retrieve
           synsets, definitions, examples, lemmas, antonyms, hypernyms,
           hyponyms, and calculate semantic similarity between words.
================================================================================
"""

import nltk
from nltk.corpus import wordnet as wn

# Ensure wordnet resources are present
try:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
except Exception:
    pass


def main():
    print("=" * 80)
    print("EXPERIMENT 17: WORDNET LEXICAL DATABASE ACCESS & EXPLORATION")
    print("=" * 80)

    target_word = "bank"
    print(f"Investigating polysemous word: '{target_word}'")

    # 1. Retrieve all synsets for target word
    synsets = wn.synsets(target_word)
    print(f"Total Synsets found for '{target_word}': {len(synsets)}\n")

    print("-" * 80)
    print(f"{'Synset ID':<20} | {'POS':<6} | {'Definition'}")
    print("-" * 80)
    for s in synsets[:6]:
        print(f"{s.name():<20} | {s.pos():<6} | {s.definition()}")
        if s.examples():
            print(f"   -> Example: \"{s.examples()[0]}\"")
    print("-" * 80)

    # 2. Explore Lemmas and Antonyms
    print("\n[2] Lemmas and Antonyms:")
    word_pairs = ["good", "hot", "increase", "love", "light"]
    for w in word_pairs:
        syn = wn.synsets(w)[0]
        lemma = syn.lemmas()[0]
        antonyms = lemma.antonyms()
        ant_names = [a.name() for a in antonyms] if antonyms else ["None found"]
        print(f"   Word: {w:<10} | Synset: {syn.name():<16} | Antonyms: {', '.join(ant_names)}")

    # 3. Lexical Hierarchy: Hypernyms, Hyponyms, and Meronyms
    print("\n[3] Taxonomic Relations (Hierarchy) for 'dog.n.01':")
    dog_syn = wn.synset("dog.n.01")
    print(f"Target Synset: {dog_syn.name()} - {dog_syn.definition()}")

    # Direct Hypernyms (Superordinate / Parent concept)
    hypernyms = dog_syn.hypernyms()
    print(f"\n * Direct Hypernyms (Parents): {[h.name() for h in hypernyms]}")

    # Root Hypernym Path
    hypernym_paths = dog_syn.hypernym_paths()
    print(f" * Taxonomic Path to Root Entity:")
    for step in hypernym_paths[0][-6:]:
        print(f"     -> {step.name()}")

    # Hyponyms (Subordinate / Child concepts)
    hyponyms = dog_syn.hyponyms()
    print(f"\n * Direct Hyponyms (Children, first 6): {[h.name() for h in hyponyms[:6]]}")

    # Part Meronyms (Components / Parts of dog)
    part_meronyms = dog_syn.part_meronyms()
    print(f" * Part Meronyms: {[m.name() for m in part_meronyms]}")

    # 4. Semantic Similarity Calculation
    print("\n[4] Semantic Similarity Between Words:")
    pairs_to_compare = [
        ("dog.n.01", "cat.n.01"),
        ("dog.n.01", "wolf.n.01"),
        ("dog.n.01", "car.n.01"),
        ("car.n.01", "automobile.n.01"),
        ("teacher.n.01", "student.n.01"),
        ("water.n.01", "fire.n.01")
    ]

    print("-" * 80)
    print(f"{'Concept 1':<18} | {'Concept 2':<18} | {'Path Sim':<10} | {'Wu-Palmer Sim'}")
    print("-" * 80)

    for s1_id, s2_id in pairs_to_compare:
        s1 = wn.synset(s1_id)
        s2 = wn.synset(s2_id)

        path_sim = s1.path_similarity(s2) or 0.0
        wup_sim = s1.wup_similarity(s2) or 0.0

        print(f"{s1.name():<18} | {s2.name():<18} | {path_sim:<10.4f} | {wup_sim:<10.4f}")
    print("-" * 80)


if __name__ == "__main__":
    main()
