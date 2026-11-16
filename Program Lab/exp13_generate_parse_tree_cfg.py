"""
================================================================================
EXPERIMENT 13
Title: Parse Tree Generation for Sentences using Context-Free Grammar (CFG)
Objective: Generate, inspect, pretty-print, and traverse parse trees for sentences
           using Context-Free Grammars in Python, demonstrating structural ambiguity.
================================================================================
"""

import nltk
from nltk import CFG, ChartParser


def main():
    print("=" * 80)
    print("EXPERIMENT 13: PARSE TREE GENERATION USING CFG")
    print("=" * 80)

    # 1. Grammar with Prepositional Phrase (PP) Attachment Ambiguity
    grammar_text = """
        S -> NP VP
        NP -> Det N | Det Adj N | NP PP | 'I'
        VP -> V NP | VP PP
        PP -> P NP
        Det -> 'the' | 'a' | 'an' | 'my'
        Adj -> 'big' | 'sleepy'
        N -> 'elephant' | 'pajamas' | 'man' | 'telescope' | 'park' | 'dog'
        V -> 'shot' | 'saw' | 'chased'
        P -> 'in' | 'with'
    """
    grammar = CFG.fromstring(grammar_text)
    parser = ChartParser(grammar)

    print("Defined Context-Free Grammar Rules:")
    for prod in grammar.productions():
        print(f"   {prod}")

    # Famous ambiguous test sentences
    test_sentences = [
        "I shot an elephant in my pajamas",
        "the man saw the dog with a telescope",
        "the sleepy dog chased a big elephant"
    ]

    for sent_idx, sentence in enumerate(test_sentences, 1):
        print("\n" + "=" * 80)
        print(f"SENTENCE #{sent_idx}: \"{sentence}\"")
        print("=" * 80)

        tokens = sentence.split()
        parse_trees = list(parser.parse(tokens))

        print(f"Total Parse Trees generated: {len(parse_trees)}")

        for tree_idx, tree in enumerate(parse_trees, 1):
            print(f"\n--- Parse Tree #{tree_idx} ---")

            # A. Bracketed / Parenthesized String representation
            print("Bracketed Representation:")
            print(" ", tree.pformat(margin=70))

            # B. Pretty ASCII Tree Diagram
            print("\nASCII Syntactic Tree:")
            tree.pretty_print()

            # C. Tree Properties and Constituents
            print(f"Tree Height: {tree.height()}")
            print(f"Total Subtrees (Constituents): {len(list(tree.subtrees()))}")
            print("Extracted Noun Phrases (NP):")
            for s in tree.subtrees(lambda t: t.label() == "NP"):
                print(f"   * {' '.join(s.leaves())}")

            print("Extracted Verb Phrases (VP):")
            for s in tree.subtrees(lambda t: t.label() == "VP"):
                print(f"   * {' '.join(s.leaves())}")


if __name__ == "__main__":
    main()
