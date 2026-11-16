"""
================================================================================
EXPERIMENT 11
Title: Top-Down Parser for Context-Free Grammars (CFG)
Objective: Implement a top-down recursive descent parser in Python for
           context-free grammars and demonstrate parsing with grammar rules
           and step-by-step expansion/backtracking.
================================================================================
"""

import nltk
from nltk import CFG, RecursiveDescentParser


def custom_top_down_parse(grammar_rules, start_symbol, tokens, verbose=True):
    """
    Self-contained recursive top-down parsing algorithm with backtracking.
    State representation: (remaining_symbols_to_expand, remaining_tokens_to_match)
    """
    step_counter = [0]

    def parse_helper(symbols, words, depth=0):
        step_counter[0] += 1
        indent = "  " * depth

        if verbose:
            print(f"{indent}[Step {step_counter[0]}] Target symbols: {symbols} | Remaining words: {words}")

        # Base case 1: Successfully derived all symbols and consumed all words
        if not symbols and not words:
            if verbose:
                print(f"{indent}==> SUCCESS: Sentence fully parsed!")
            return True, ["()"]

        # Base case 2: Ran out of symbols or words prematurely
        if not symbols or not words:
            if verbose:
                print(f"{indent}--> Backtrack: Symbol/word mismatch.")
            return False, []

        first_sym = symbols[0]
        rest_syms = symbols[1:]

        # Case A: first symbol is a terminal (string literal)
        if first_sym not in grammar_rules:
            if first_sym == words[0]:
                if verbose:
                    print(f"{indent}--> Match terminal '{first_sym}' with word '{words[0]}'")
                matched, subtrees = parse_helper(rest_syms, words[1:], depth + 1)
                if matched:
                    return True, [first_sym] + subtrees
            else:
                if verbose:
                    print(f"{indent}--> Mismatch: Expected terminal '{first_sym}', got '{words[0]}'")
            return False, []

        # Case B: first symbol is a non-terminal (expand using grammar rules)
        productions = grammar_rules.get(first_sym, [])
        for prod in productions:
            if verbose:
                print(f"{indent}--> Expand Non-Terminal '{first_sym}' -> {' '.join(prod)}")
            new_symbols = list(prod) + rest_syms
            matched, subtrees = parse_helper(new_symbols, words, depth + 1)
            if matched:
                return True, [(first_sym, prod, subtrees)]

        if verbose:
            print(f"{indent}--> Backtrack from '{first_sym}': All expansions exhausted.")
        return False, []

    return parse_helper([start_symbol], tokens)


def main():
    print("=" * 75)
    print("EXPERIMENT 11: TOP-DOWN PARSER FOR CONTEXT-FREE GRAMMARS (CFG)")
    print("=" * 75)

    # 1. Define Context-Free Grammar
    cfg_text = """
        S -> NP VP
        NP -> Det N | Det Adj N | 'John' | 'Mary'
        VP -> V NP | V NP PP | V
        PP -> P NP
        Det -> 'the' | 'a'
        Adj -> 'smart' | 'lazy'
        N -> 'dog' | 'cat' | 'telescope' | 'park' | 'student'
        V -> 'saw' | 'chased' | 'likes'
        P -> 'in' | 'with'
    """
    grammar = CFG.fromstring(cfg_text)

    print("Defined Context-Free Grammar:")
    for prod in grammar.productions():
        print(f"   {prod}")

    # Build dictionary rules for custom parser
    rules_dict = {}
    for prod in grammar.productions():
        lhs = str(prod.lhs())
        rhs = [str(sym) for sym in prod.rhs()]
        rules_dict.setdefault(lhs, []).append(rhs)

    # 2. Test Sentences
    test_sentences = [
        "the smart dog chased a cat",
        "John saw the student with a telescope",
        "the cat chased",
        "the dog Mary"  # Invalid sentence
    ]

    # Initialize NLTK's RecursiveDescentParser
    rd_parser = RecursiveDescentParser(grammar)

    for i, sent in enumerate(test_sentences, 1):
        print("\n" + "=" * 75)
        print(f"PARSING SENTENCE #{i}: \"{sent}\"")
        print("=" * 75)

        tokens = sent.split()

        # Step-by-step custom recursive descent trace
        print("\n[A] Top-Down Recursive Expansion Trace:")
        success, tree_data = custom_top_down_parse(rules_dict, "S", tokens, verbose=True)

        print(f"\nParse Result: {'VALID (Grammatically Accepted)' if success else 'INVALID (Rejected)'}")

        # NLTK parse tree generation
        if success:
            print("\n[B] Generated NLTK Parse Trees:")
            try:
                trees = list(rd_parser.parse(tokens))
                for t_idx, tree in enumerate(trees, 1):
                    print(f"\nTree #{t_idx}:")
                    tree.pretty_print()
            except Exception as e:
                print(f"NLTK Parser note: {e}")


if __name__ == "__main__":
    main()
