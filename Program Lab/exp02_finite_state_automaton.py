"""
================================================================================
EXPERIMENT 02
Title: Finite State Automaton (FSA) for Pattern Recognition
Objective: Implement a Deterministic Finite Automaton (DFA) in Python that
           recognizes strings ending with the pattern 'ab'.
================================================================================
"""


class EndsWithABAutomaton:
    """
    Deterministic Finite Automaton (DFA) to recognize strings ending with 'ab'.

    Formal Definition:
      M = (Q, Sigma, delta, q0, F)
      Q (States): {'q0', 'q1', 'q2'}
        - q0: Initial state (suffix does not end in 'a' or 'ab')
        - q1: State representing last character was 'a'
        - q2: State representing last two characters were 'ab' (Accepting state)
      Sigma (Alphabet): characters a, b (and other characters reset to q0)
      q0 (Start State): 'q0'
      F (Accepting States): {'q2'}
    """

    def __init__(self):
        self.states = {"q0", "q1", "q2"}
        self.start_state = "q0"
        self.accept_states = {"q2"}

        # Transition table: (current_state, symbol) -> next_state
        self.transitions = {
            ("q0", "a"): "q1",
            ("q0", "b"): "q0",
            ("q1", "a"): "q1",
            ("q1", "b"): "q2",
            ("q2", "a"): "q1",
            ("q2", "b"): "q0",
        }

    def transition(self, current_state: str, symbol: str) -> str:
        """Determines the next state given current state and input symbol."""
        # Any symbol other than 'a' and 'b' resets to q0
        return self.transitions.get((current_state, symbol), "q0")

    def process(self, input_string: str, verbose: bool = False):
        """
        Processes an input string symbol by symbol and returns whether
        the automaton accepts it, along with the execution trace.
        """
        current_state = self.start_state
        trace = [current_state]

        if verbose:
            print(f"\nTracing input: '{input_string}'")
            print(f"  Start State: {current_state}")

        for idx, char in enumerate(input_string):
            next_state = self.transition(current_state, char)
            if verbose:
                print(f"  Step {idx + 1}: State '{current_state}' --('{char}')--> State '{next_state}'")
            current_state = next_state
            trace.append(current_state)

        is_accepted = current_state in self.accept_states
        if verbose:
            print(f"  Final State: '{current_state}' -> Accepted? {'YES' if is_accepted else 'NO'}")

        return is_accepted, current_state, trace

    def print_transition_table(self):
        """Prints the state transition table."""
        print("\nState Transition Table (delta):")
        print("+---------------+---------+---------+-------------------+")
        print("| Current State | Input 'a' | Input 'b' | Input (other)     |")
        print("+---------------+---------+---------+-------------------+")
        print("| -> q0 (Start) |   q1    |   q0    |        q0         |")
        print("|    q1         |   q1    |   q2    |        q0         |")
        print("|  * q2 (Accept)|   q1    |   q0    |        q0         |")
        print("+---------------+---------+---------+-------------------+")
        print("Legend: -> = Start State, * = Accepting State\n")


def main():
    print("=" * 70)
    print("EXPERIMENT 02: FINITE STATE AUTOMATON (STRINGS ENDING WITH 'ab')")
    print("=" * 70)

    fsa = EndsWithABAutomaton()
    fsa.print_transition_table()

    # Test cases: positive and negative examples
    test_cases = [
        "ab",          # Valid
        "aab",         # Valid
        "bab",         # Valid
        "abab",        # Valid
        "xyzab",       # Valid
        "aba",         # Invalid (ends with 'ba')
        "b",           # Invalid
        "a",           # Invalid
        "abb",         # Invalid
        "",            # Invalid (empty string)
        "cab",         # Valid
        "tabc",        # Invalid
    ]

    print("Running Test Suite:")
    print("-" * 70)
    print(f"{'Input String':<15} | {'Final State':<12} | {'Accepted?':<10} | {'Expected'}")
    print("-" * 70)

    for test in test_cases:
        accepted, final_state, _ = fsa.process(test, verbose=False)
        expected = "ACCEPTED" if test.endswith("ab") else "REJECTED"
        result_str = "ACCEPTED" if accepted else "REJECTED"
        status = "PASS" if result_str == expected else "FAIL"
        print(f"'{test}':{'' if len(test)>10 else ' '*(13-len(test))} | {final_state:<12} | {result_str:<10} | {expected} [{status}]")

    print("-" * 70)
    # Detailed trace for demonstration
    print("\nDetailed Trace of Positive and Negative Samples:")
    fsa.process("cab", verbose=True)
    fsa.process("aba", verbose=True)


if __name__ == "__main__":
    main()
