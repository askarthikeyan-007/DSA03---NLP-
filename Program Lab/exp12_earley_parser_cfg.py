"""
================================================================================
EXPERIMENT 12
Title: Earley Parser for Context-Free Grammars (CFG)
Objective: Implement an Earley chart parser in Python for Context-Free Grammars,
           demonstrating Predictor, Scanner, and Completer operations.
================================================================================
"""

import nltk
from nltk import CFG
from nltk.parse.earleychart import EarleyChartParser


class EarleyState:
    """Represents an Earley item/state: A -> alpha . beta [origin]"""

    def __init__(self, lhs: str, rhs: tuple, dot: int, origin: int):
        self.lhs = lhs
        self.rhs = rhs
        self.dot = dot
        self.origin = origin

    def next_symbol(self):
        """Returns the symbol immediately following the dot, or None if complete."""
        if self.dot < len(self.rhs):
            return self.rhs[self.dot]
        return None

    def is_complete(self):
        """Checks if the dot has reached the end of the RHS."""
        return self.dot >= len(self.rhs)

    def advance(self):
        """Creates a new state with dot advanced by 1 position."""
        return EarleyState(self.lhs, self.rhs, self.dot + 1, self.origin)

    def __eq__(self, other):
        return (
            isinstance(other, EarleyState)
            and self.lhs == other.lhs
            and self.rhs == other.rhs
            and self.dot == other.dot
            and self.origin == other.origin
        )

    def __hash__(self):
        return hash((self.lhs, self.rhs, self.dot, self.origin))

    def __repr__(self):
        before_dot = " ".join(self.rhs[:self.dot])
        after_dot = " ".join(self.rhs[self.dot:])
        return f"{self.lhs} -> {before_dot} . {after_dot} [{self.origin}]"


class EarleyParser:
    """
    Earley Parser Implementation.
    Three Fundamental Operations:
      1. Predictor: Adds candidate rules for non-terminals.
      2. Scanner: Reads matching input terminal words from text.
      3. Completer: Propagates completed sub-trees back to parent states.
    """

    def __init__(self, grammar_productions, start_symbol="S"):
        self.productions = grammar_productions
        self.start_symbol = start_symbol

        # Group productions by LHS
        self.rules = {}
        for lhs, rhs in self.productions:
            self.rules.setdefault(lhs, []).append(rhs)

    def parse(self, words, verbose=True):
        n = len(words)
        chart = [[] for _ in range(n + 1)]
        chart_sets = [set() for _ in range(n + 1)]

        def add_state(state, chart_index, op_name):
            if state not in chart_sets[chart_index]:
                chart_sets[chart_index].add(state)
                chart[chart_index].append((state, op_name))

        # Initial dummy state: GAMMA -> . S [0]
        gamma_state = EarleyState("GAMMA", (self.start_symbol,), 0, 0)
        add_state(gamma_state, 0, "INIT")

        for i in range(n + 1):
            if verbose:
                print(f"\n--- Chart [{i}] (Input word: '{words[i-1] if i > 0 else '<START>'}') ---")

            state_idx = 0
            while state_idx < len(chart[i]):
                state, op = chart[i][state_idx]
                state_idx += 1

                if not state.is_complete():
                    next_sym = state.next_symbol()
                    # Check if next_sym is a non-terminal (Predictor)
                    if next_sym in self.rules:
                        for prod_rhs in self.rules[next_sym]:
                            new_state = EarleyState(next_sym, prod_rhs, 0, i)
                            add_state(new_state, i, "PREDICTOR")
                    # Next_sym is a terminal (Scanner)
                    elif i < n and next_sym == words[i]:
                        new_state = state.advance()
                        add_state(new_state, i + 1, f"SCANNER('{words[i]}')")
                else:
                    # State is complete (Completer)
                    completed_lhs = state.lhs
                    origin = state.origin
                    for prev_state, _ in chart[origin]:
                        if not prev_state.is_complete() and prev_state.next_symbol() == completed_lhs:
                            new_state = prev_state.advance()
                            add_state(new_state, i, f"COMPLETER({completed_lhs})")

            if verbose:
                for s, op_name in chart[i]:
                    print(f"   {repr(s):<36} | Source: {op_name}")

        # Check if GAMMA -> S . [0] is in chart[n]
        accept_state = EarleyState("GAMMA", (self.start_symbol,), 1, 0)
        is_accepted = accept_state in chart_sets[n]
        return is_accepted, chart


def main():
    print("=" * 80)
    print("EXPERIMENT 12: EARLEY PARSER FOR CONTEXT-FREE GRAMMARS")
    print("=" * 80)

    # Grammar definition: (LHS, (RHS_tuple))
    productions = [
        ("S", ("NP", "VP")),
        ("NP", ("Det", "N")),
        ("NP", ("NP", "PP")),
        ("VP", ("V", "NP")),
        ("VP", ("VP", "PP")),
        ("PP", ("P", "NP")),
        ("Det", ("the",)),
        ("Det", ("a",)),
        ("N", ("boy",)),
        ("N", ("telescope",)),
        ("N", ("girl",)),
        ("V", ("saw",)),
        ("P", ("with",)),
    ]

    print("Grammar Productions:")
    for lhs, rhs in productions:
        print(f"   {lhs} -> {' '.join(rhs)}")

    parser = EarleyParser(productions, start_symbol="S")

    sentence = "the boy saw a girl with a telescope"
    words = sentence.split()

    print("\n" + "=" * 80)
    print(f"EARLEY PARSING OF SENTENCE: \"{sentence}\"")
    print("=" * 80)

    accepted, chart = parser.parse(words, verbose=True)

    print("\n" + "=" * 80)
    print(f"Final Earley Parsing Status: {'ACCEPTED (Grammatically Valid)' if accepted else 'REJECTED'}")
    print("=" * 80)

    # Compare with NLTK's EarleyChartParser
    print("\n[Verification using NLTK EarleyChartParser]:")
    cfg_grammar = CFG.fromstring("""
        S -> NP VP
        NP -> Det N | NP PP
        VP -> V NP | VP PP
        PP -> P NP
        Det -> 'the' | 'a'
        N -> 'boy' | 'telescope' | 'girl'
        V -> 'saw'
        P -> 'with'
    """)
    nltk_earley = EarleyChartParser(cfg_grammar)
    trees = list(nltk_earley.parse(words))
    print(f"Total Parse Trees Found: {len(trees)}")
    for idx, t in enumerate(trees, 1):
        print(f"\nParse Tree #{idx}:")
        t.pretty_print()


if __name__ == "__main__":
    main()
