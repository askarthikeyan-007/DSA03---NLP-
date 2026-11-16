"""
================================================================================
EXPERIMENT 14
Title: Agreement Checking in Sentences using Feature-Based CFG
Objective: Implement a program in Python to check Subject-Verb and Determiner-Noun
           number agreement (singular vs plural) based on Context-Free Grammar rules.
================================================================================
"""

import nltk
from nltk import grammar, parse


def build_feature_grammar():
    """
    Constructs an NLTK Feature Grammar with [NUM=?n] agreement unification.
    ?n can be 'sg' (singular) or 'pl' (plural).
    """
    fgrammar_str = """
        % start S
        # Sentence rule enforcing Subject NP and VP agreement in Number (?n)
        S -> NP[NUM=?n] VP[NUM=?n]

        # Noun Phrase enforcing Determiner and Noun agreement in Number (?n)
        NP[NUM=?n] -> Det[NUM=?n] N[NUM=?n]
        NP[NUM=?n] -> Det[NUM=any] N[NUM=?n]
        NP[NUM=?n] -> PropN[NUM=?n]

        # Verb Phrase matching Verb number (?n)
        VP[NUM=?n] -> V[NUM=?n] NP[NUM=?m]
        VP[NUM=?n] -> V[NUM=?n]

        # Lexicon with morphological features
        Det[NUM=sg] -> 'this' | 'a' | 'every'
        Det[NUM=pl] -> 'these' | 'those' | 'many'
        Det[NUM=any] -> 'the'

        PropN[NUM=sg] -> 'John' | 'Mary'

        N[NUM=sg] -> 'dog' | 'cat' | 'student' | 'boy'
        N[NUM=pl] -> 'dogs' | 'cats' | 'students' | 'boys'

        V[NUM=sg] -> 'barks' | 'chases' | 'likes' | 'runs'
        V[NUM=pl] -> 'bark' | 'chase' | 'like' | 'run'
    """
    return grammar.FeatureGrammar.fromstring(fgrammar_str)


def inspect_agreement_manually(tokens):
    """
    Programmatic feature agreement validator that reports exact linguistic discrepancies.
    """
    singular_nouns = {"dog", "cat", "student", "boy", "john", "mary"}
    plural_nouns = {"dogs", "cats", "students", "boys"}

    singular_verbs = {"barks", "chases", "likes", "runs", "is"}
    plural_verbs = {"bark", "chase", "like", "run", "are"}

    singular_dets = {"this", "a", "every"}
    plural_dets = {"these", "those", "many"}

    reasons = []

    # Check Det + Noun agreement
    if len(tokens) >= 2:
        d, n = tokens[0].lower(), tokens[1].lower()
        if d in singular_dets and n in plural_nouns:
            reasons.append(f"Determiner-Noun mismatch: '{d}' (singular) cannot modify '{n}' (plural).")
        elif d in plural_dets and n in singular_nouns:
            reasons.append(f"Determiner-Noun mismatch: '{d}' (plural) cannot modify '{n}' (singular).")

    # Check Subject-Verb agreement
    subject_num = None
    verb_num = None

    for w in tokens:
        wl = w.lower()
        if wl in singular_nouns:
            subject_num = "singular"
        elif wl in plural_nouns:
            subject_num = "plural"

        if wl in singular_verbs:
            verb_num = "singular"
        elif wl in plural_verbs:
            verb_num = "plural"

    if subject_num and verb_num:
        if subject_num != verb_num:
            reasons.append(
                f"Subject-Verb agreement violation: Subject is {subject_num.upper()} but verb is {verb_num.upper()}."
            )

    is_valid = len(reasons) == 0
    return is_valid, reasons


def main():
    print("=" * 80)
    print("EXPERIMENT 14: SYNTACTIC AGREEMENT CHECKING USING FEATURE CFG")
    print("=" * 80)

    fgrammar = build_feature_grammar()
    parser = parse.FeatureEarleyChartParser(fgrammar)

    test_sentences = [
        # Grammatically Correct
        "the dog barks",
        "the dogs bark",
        "this dog barks",
        "these dogs bark",
        "John likes dogs",
        # Agreement Violations
        "the dog bark",          # Subject (sg) != Verb (pl)
        "the dogs barks",         # Subject (pl) != Verb (sg)
        "these dog barks",        # Det (pl) != Noun (sg)
        "this dogs bark",         # Det (sg) != Noun (pl)
        "this dog bark"           # Both Det-Noun ok, but Subj (sg) != Verb (pl)
    ]

    print(f"{'Sentence':<24} | {'Grammar Parse':<15} | {'Agreement Status'}")
    print("-" * 80)

    for sentence in test_sentences:
        tokens = sentence.split()
        trees = list(parser.parse(tokens))
        has_parse = len(trees) > 0

        is_valid, reasons = inspect_agreement_manually(tokens)
        status_str = "VALID AGREEMENT" if has_parse else "AGREEMENT VIOLATION"

        print(f"\"{sentence}\":{'' if len(sentence) > 20 else ' '*(22 - len(sentence))} | {len(trees)} tree(s) found  | {status_str}")
        if reasons:
            for r in reasons:
                print(f"   -> Diagnostic: {r}")

    print("-" * 80)
    print("\nDetailed Feature Tree for Valid Sentence 'this dog barks':")
    valid_trees = list(parser.parse("this dog barks".split()))
    if valid_trees:
        valid_trees[0].pretty_print()


if __name__ == "__main__":
    main()
