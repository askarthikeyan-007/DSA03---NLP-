"""
================================================================================
EXPERIMENT 09
Title: Rule-Based Part-of-Speech Tagging using Regular Expressions
Objective: Implement a rule-based POS tagging system in Python using regular
           expression patterns and evaluate its behavior on diverse sentences.
================================================================================
"""

import nltk
from nltk.tag import RegexpTagger

# Ensure tokenizers are available
try:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
except Exception:
    pass

# Ordered Regular Expression Rules for POS Tagging
# Priority matters: Specific patterns are checked before general patterns.
REGEX_RULES = [
    # Numbers
    (r"^-?[0-9]+(\.[0-9]+)?$", "CD"),       # Cardinal numbers (e.g. 42, 3.14)
    # Common function words
    (r"^(the|a|an|this|that|these|those)$", "DT"),   # Determiners
    (r"^(in|on|at|by|for|with|about|against|between|into|through|during|before|after|above|below|to|from|up|down|in|out|off|over|under)$", "IN"), # Prepositions
    (r"^(and|but|or|nor|so|yet)$", "CC"),    # Conjunctions
    (r"^(i|you|he|she|it|we|they|me|him|her|us|them)$", "PRP"), # Personal Pronouns
    (r"^(my|your|his|her|its|our|their)$", "PRP$"),  # Possessive Pronouns
    (r"^(can|could|may|might|must|shall|should|will|would)$", "MD"), # Modals
    (r"^(is|am|are|was|were|be|been|being)$", "VB"), # Be-verbs
    # Suffix-based morphological rules
    (r".*ing$", "VBG"),                      # Gerunds / present participles (e.g. running)
    (r".*ed$", "VBD"),                       # Past tense verbs (e.g. walked)
    (r".*es$", "VBZ"),                       # 3rd person singular verbs (e.g. washes)
    (r".*'s$", "POS"),                       # Possessive nouns (e.g. John's)
    (r".*ly$", "RB"),                        # Adverbs (e.g. quickly)
    (r".*able$", "JJ"),                      # Adjectives (e.g. comfortable)
    (r".*ful$", "JJ"),                       # Adjectives (e.g. beautiful)
    (r".*ous$", "JJ"),                       # Adjectives (e.g. dangerous)
    (r".*al$", "JJ"),                        # Adjectives (e.g. natural)
    (r".*ive$", "JJ"),                       # Adjectives (e.g. active)
    (r".*ment$", "NN"),                      # Nouns (e.g. development)
    (r".*ness$", "NN"),                      # Nouns (e.g. darkness)
    (r".*tion$", "NN"),                      # Nouns (e.g. action)
    (r".*sion$", "NN"),                      # Nouns (e.g. decision)
    (r".*er$", "NN"),                        # Agent Nouns (e.g. programmer)
    (r".*s$", "NNS"),                        # Plural nouns (e.g. books, algorithms)
    (r"^[A-Z][a-z]+$", "NNP"),               # Proper nouns capitalized (e.g. Alice, Paris)
    (r"^[.,!?;:]$", "."),                    # Punctuation marks
    # Default fallback rule
    (r".*", "NN"),                           # Default noun tag
]


class CustomRegexPOSTagger:
    """A self-contained Regex-based POS Tagger showing matching logic."""

    def __init__(self, rules):
        self.rules = rules
        self.nltk_tagger = RegexpTagger(rules)

    def tag_with_trace(self, tokens):
        """Tags tokens and reports which regex rule matched each token."""
        results = []
        for word in tokens:
            matched_rule = None
            assigned_tag = "NN"
            for pattern, tag in self.rules:
                import re
                if re.match(pattern, word, re.IGNORECASE if not pattern.startswith("^[A-Z]") else 0):
                    assigned_tag = tag
                    matched_rule = pattern
                    break
            results.append((word, assigned_tag, matched_rule))
        return results


def main():
    print("=" * 80)
    print("EXPERIMENT 09: RULE-BASED POS TAGGING USING REGULAR EXPRESSIONS")
    print("=" * 80)

    tagger = CustomRegexPOSTagger(REGEX_RULES)

    print(f"Total Regular Expression Rules configured: {len(REGEX_RULES)}")
    print("\nSample Rules:")
    for pat, tag in REGEX_RULES[:8]:
        print(f"   Pattern: {pat:<35} -> Tag: {tag}")
    print("   ... [morphological suffix rules] ...")
    print(f"   Pattern: {REGEX_RULES[-1][0]:<35} -> Tag: {REGEX_RULES[-1][1]} (Fallback)")

    test_sentences = [
        "The quick brown fox jumped gracefully over 12 lazy sleeping dogs .",
        "Alice created a beautiful computational model and carefully tested its performance .",
        "Modern programmers would build scalable applications with wonderful readability ."
    ]

    print("\n" + "=" * 80)
    print("EVALUATION ON TEST SENTENCES")
    print("=" * 80)

    for i, sent in enumerate(test_sentences, 1):
        print(f"\n[Sentence #{i}]: \"{sent}\"")
        tokens = nltk.word_tokenize(sent)
        tagged_trace = tagger.tag_with_trace(tokens)

        print("-" * 80)
        print(f"{'Token':<16} | {'Assigned Tag':<14} | {'Matched Regex Pattern'}")
        print("-" * 80)
        for word, tag, pattern in tagged_trace:
            print(f"{word:<16} | {tag:<14} | {pattern}")
        print("-" * 80)


if __name__ == "__main__":
    main()
