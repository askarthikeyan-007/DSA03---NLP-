"""
================================================================================
EXPERIMENT 15
Title: Probabilistic Context-Free Grammar (PCFG) Parsing
Objective: Implement a Probabilistic Context-Free Grammar (PCFG) in Python,
           calculate parse tree probabilities, and use the Viterbi PCFG parser
           to disambiguate syntactically ambiguous sentences.
================================================================================
"""

import nltk
from nltk import PCFG, ViterbiParser, InsideChartParser
import math


def main():
    print("=" * 80)
    print("EXPERIMENT 15: PROBABILISTIC CONTEXT-FREE GRAMMAR (PCFG) PARSING")
    print("=" * 80)

    # 1. Define PCFG Grammar with Rule Probabilities
    pcfg_text = """
        S -> NP VP [1.0]
        VP -> V NP [0.5] | VP PP [0.3] | V [0.2]
        NP -> Det N [0.4] | NP PP [0.3] | Det Adj N [0.2] | 'astronomers' [0.1]
        PP -> P NP [1.0]
        Det -> 'the' [0.6] | 'a' [0.4]
        Adj -> 'bright' [0.7] | 'distant' [0.3]
        N -> 'star' [0.45] | 'telescope' [0.35] | 'astronomer' [0.2]
        V -> 'saw' [1.0]
        P -> 'with' [1.0]
    """
    grammar = PCFG.fromstring(pcfg_text)

    print("Defined PCFG Productions and Probabilities:")
    for prod in grammar.productions():
        print(f"   {prod}")

    # 2. Viterbi Parser (Finds Most Probable Parse Tree)
    viterbi_parser = ViterbiParser(grammar)
    # InsideChartParser finds all possible parse trees with their probabilities
    chart_parser = InsideChartParser(grammar)

    test_sentence = "the astronomer saw the star with a telescope"
    tokens = test_sentence.split()

    print("\n" + "=" * 80)
    print(f"PARSING AMBIGUOUS SENTENCE: \"{test_sentence}\"")
    print("=" * 80)

    # Find all possible parse trees
    all_trees = list(chart_parser.parse(tokens))
    print(f"Total Syntactic Parse Trees generated: {len(all_trees)}")

    # Sort trees by their probability descending
    all_trees.sort(key=lambda t: t.prob(), reverse=True)

    for idx, tree in enumerate(all_trees, 1):
        prob = tree.prob()
        log_prob = math.log(prob) if prob > 0 else -float("inf")
        print(f"\n--- Parse Tree #{idx} [Probability = {prob:.8e} | Log Prob = {log_prob:.4f}] ---")

        # Interpretation of PP attachment
        if "VP -> VP PP" in str(tree.productions()):
            interpretation = "Verb Attachment: The astronomer used the telescope to see the star."
        else:
            interpretation = "Noun Attachment: The astronomer saw a star that was accompanied by a telescope."

        print(f"Linguistic Interpretation: {interpretation}")
        print("\nBracketed Tree:")
        print(" ", tree.pformat(margin=70))
        print("\nTree Diagram:")
        tree.pretty_print()

    # 3. Viterbi Best Parse Selection
    print("=" * 80)
    print("VITERBI OPTIMAL PARSE TREE (Argmax)")
    print("=" * 80)
    best_parse = list(viterbi_parser.parse(tokens))
    if best_parse:
        best_tree = best_parse[0]
        print(f"Best Tree Probability : {best_tree.prob():.8e}")
        best_tree.pretty_print()


if __name__ == "__main__":
    main()
