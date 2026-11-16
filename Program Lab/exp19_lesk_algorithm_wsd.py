"""
================================================================================
EXPERIMENT 19
Title: Word Sense Disambiguation (WSD) using Lesk Algorithm
Objective: Implement both custom Simplified Lesk and NLTK Lesk algorithms
           in Python to resolve word sense ambiguity based on context overlap.
================================================================================
"""

import re
import nltk
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk as nltk_lesk

# Ensure required resources are available
try:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = {"the", "a", "an", "in", "on", "of", "and", "or", "is", "was", "to", "for", "with", "at", "by"}


def clean_tokenize(text: str):
    """Converts text to lower-case word tokens excluding punctuation and stopwords."""
    words = re.findall(r"\b[a-z]{2,}\b", text.lower())
    return {w for w in words if w not in STOPWORDS}


def simplified_lesk(word: str, sentence: str):
    """
    Custom implementation of Simplified Lesk Algorithm:
      1. Collect candidate synsets for the target word.
      2. Construct signature for each synset: definition + examples.
      3. Compute overlap between context words and signature words.
      4. Return synset with highest overlap score.
    """
    context_tokens = clean_tokenize(sentence)
    synsets = wn.synsets(word)

    if not synsets:
        return None, 0, []

    best_synset = synsets[0]
    max_overlap = -1
    overlap_details = []

    for s in synsets:
        # Signature: definition + example sentences
        signature_text = s.definition() + " " + " ".join(s.examples())
        signature_tokens = clean_tokenize(signature_text)

        # Overlap computation
        shared_words = context_tokens.intersection(signature_tokens)
        score = len(shared_words)

        overlap_details.append({
            "synset": s.name(),
            "score": score,
            "shared": sorted(list(shared_words)),
            "definition": s.definition()
        })

        if score > max_overlap:
            max_overlap = score
            best_synset = s

    # Sort candidates by overlap score descending
    overlap_details.sort(key=lambda x: x["score"], reverse=True)
    return best_synset, max_overlap, overlap_details


def main():
    print("=" * 80)
    print("EXPERIMENT 19: WORD SENSE DISAMBIGUATION (LESK ALGORITHM)")
    print("=" * 80)

    test_scenarios = [
        ("bank", "The financial bank approved my business loan and managed my money deposit."),
        ("bank", "We sat on the muddy river bank enjoying the calm flowing water and trees."),
        ("bass", "The fisherman caught a fresh sea bass using fishing bait in the lake."),
        ("bass", "The musician played electric bass guitar solos in the jazz band concert.")
    ]

    for idx, (word, sentence) in enumerate(test_scenarios, 1):
        print(f"\n[Case #{idx}]: Target Word: '{word}'")
        print(f"Sentence: \"{sentence}\"")
        print("-" * 80)

        # 1. Custom Simplified Lesk
        best_s, best_score, details = simplified_lesk(word, sentence)

        print(f"Custom Lesk Best Sense : {best_s.name()} (Overlap Score: {best_score})")
        print(f"Definition             : {best_s.definition()}")

        # Display top 3 candidate scores
        print("Candidate Sense Overlaps:")
        for cand in details[:3]:
            shared_str = ", ".join(cand["shared"]) if cand["shared"] else "None"
            print(f"   * {cand['synset']:<16} (Score: {cand['score']}) -> Overlapping tokens: [{shared_str}]")

        # 2. Compare with NLTK built-in lesk
        try:
            tokens = nltk.word_tokenize(sentence)
            nltk_res = nltk_lesk(tokens, word)
            if nltk_res:
                print(f"NLTK Lesk Result       : {nltk_res.name()} -> {nltk_res.definition()}")
        except Exception as e:
            print(f"NLTK Lesk note: {e}")


if __name__ == "__main__":
    main()
