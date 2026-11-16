"""
================================================================================
EXPERIMENT 04
Title: Finite-State Machine for Morphological Parsing (Plural English Nouns)
Objective: Implement a Finite State Machine (FSM / Morphological Transducer)
           to parse and generate plural forms of English nouns.
================================================================================
"""


class PluralNounFSM:
    """
    Finite-State Machine for English Noun Pluralization.

    States:
      - Q0: START
      - Q1: IRREGULAR_CHECK
      - Q2: SIBILANT_STATE (ends with s, x, z, ch, sh -> add 'es')
      - Q3: Y_ENDING_STATE (consonant + y -> drop y, add 'ies'; vowel + y -> add 's')
      - Q4: F_FE_ENDING_STATE (ends with f/fe -> drop f/fe, add 'ves')
      - Q5: O_ENDING_STATE (consonant + o -> add 'es'; vowel + o -> add 's')
      - Q6: DEFAULT_STATE (add 's')
      - Q_ACCEPT: Pluralized word output
    """

    IRREGULAR_MAP = {
        "child": "children",
        "man": "men",
        "woman": "women",
        "tooth": "teeth",
        "foot": "feet",
        "mouse": "mice",
        "goose": "geese",
        "ox": "oxen",
        "person": "people",
        "sheep": "sheep",
        "deer": "deer",
        "fish": "fish",
        "cactus": "cacti",
        "nucleus": "nuclei",
    }

    F_EXCEPTIONS = {"roof": "roofs", "chief": "chiefs", "cliff": "cliffs", "belief": "beliefs", "chef": "chefs"}
    O_EXCEPTIONS = {"photo": "photos", "piano": "pianos", "halo": "halos", "auto": "autos"}

    def __init__(self):
        self.vowels = set("aeiou")

    def parse_plural(self, noun: str, verbose: bool = False):
        """Processes a singular noun through the FSM to produce its plural."""
        original = noun.strip()
        lower = original.lower()
        if not lower:
            return "", ["START -> EMPTY -> REJECT"]

        trace = ["Q0 (START)"]

        # Step 1: Check Irregulars (Q0 -> Q1)
        trace.append(f"Q1 (IRREGULAR_CHECK: '{lower}')")
        if lower in self.IRREGULAR_MAP:
            result = self.IRREGULAR_MAP[lower]
            trace.append(f"Q_ACCEPT: Matched irregular lexicon -> '{result}'")
            return self._preserve_case(original, result), trace

        # Step 2: Sibilants: -s, -x, -z, -ch, -sh -> add 'es' (Q2)
        if lower.endswith(("ch", "sh")) or (len(lower) >= 1 and lower[-1] in "sxz"):
            trace.append("Q2 (SIBILANT_ENDING: ends with s/x/z/ch/sh)")
            result = lower + "es"
            trace.append(f"Q_ACCEPT: Rule +es applied -> '{result}'")
            return self._preserve_case(original, result), trace

        # Step 3: Ends in 'y' (Q3)
        if lower.endswith("y") and len(lower) > 1:
            trace.append("Q3 (Y_ENDING)")
            preceding_char = lower[-2]
            if preceding_char not in self.vowels:
                # Consonant + y -> ies
                trace.append(f"Q3a: Preceding char '{preceding_char}' is CONSONANT -> drop 'y' + 'ies'")
                result = lower[:-1] + "ies"
            else:
                # Vowel + y -> s
                trace.append(f"Q3b: Preceding char '{preceding_char}' is VOWEL -> add 's'")
                result = lower + "s"
            trace.append(f"Q_ACCEPT: Output -> '{result}'")
            return self._preserve_case(original, result), trace

        # Step 4: Ends in 'f' or 'fe' (Q4)
        if lower in self.F_EXCEPTIONS:
            trace.append(f"Q4_EXC: Common exception noun '{lower}' -> add 's'")
            result = self.F_EXCEPTIONS[lower]
            trace.append(f"Q_ACCEPT: Output -> '{result}'")
            return self._preserve_case(original, result), trace

        if lower.endswith("fe"):
            trace.append("Q4a (FE_ENDING) -> drop 'fe' + 'ves'")
            result = lower[:-2] + "ves"
            trace.append(f"Q_ACCEPT: Output -> '{result}'")
            return self._preserve_case(original, result), trace
        elif lower.endswith("f"):
            trace.append("Q4b (F_ENDING) -> drop 'f' + 'ves'")
            result = lower[:-1] + "ves"
            trace.append(f"Q_ACCEPT: Output -> '{result}'")
            return self._preserve_case(original, result), trace

        # Step 5: Ends in 'o' (Q5)
        if lower.endswith("o") and len(lower) > 1:
            trace.append("Q5 (O_ENDING)")
            if lower in self.O_EXCEPTIONS:
                result = self.O_EXCEPTIONS[lower]
                trace.append(f"Q5_EXC: Output -> '{result}'")
                return self._preserve_case(original, result), trace
            preceding = lower[-2]
            if preceding not in self.vowels:
                result = lower + "es"
                trace.append(f"Q5a: Consonant + 'o' -> add 'es' -> '{result}'")
            else:
                result = lower + "s"
                trace.append(f"Q5b: Vowel + 'o' -> add 's' -> '{result}'")
            trace.append(f"Q_ACCEPT: Output -> '{result}'")
            return self._preserve_case(original, result), trace

        # Step 6: Default (Q6) -> add 's'
        trace.append("Q6 (DEFAULT_ENDING) -> append 's'")
        result = lower + "s"
        trace.append(f"Q_ACCEPT: Output -> '{result}'")
        return self._preserve_case(original, result), trace

    def _preserve_case(self, original: str, output: str) -> str:
        if original.isupper():
            return output.upper()
        elif original.istitle():
            return output.capitalize()
        return output


def main():
    print("=" * 75)
    print("EXPERIMENT 04: FSM FOR MORPHOLOGICAL PARSING (ENGLISH PLURALS)")
    print("=" * 75)

    fsm = PluralNounFSM()

    test_nouns = [
        # Irregulars
        "child", "man", "woman", "foot", "tooth", "mouse", "sheep",
        # Sibilants
        "bus", "box", "buzz", "church", "dish", "watch",
        # Y endings
        "baby", "city", "fly", "party", "boy", "day", "key",
        # F / FE endings
        "leaf", "knife", "wolf", "life", "roof", "chief",
        # O endings
        "hero", "potato", "tomato", "echo", "zoo", "radio", "piano",
        # Default
        "book", "cat", "dog", "computer", "pen"
    ]

    print("-" * 75)
    print(f"{'Singular Noun':<18} | {'Generated Plural':<20} | {'Morphological Rule'}")
    print("-" * 75)

    for noun in test_nouns:
        plural, trace = fsm.parse_plural(noun)
        # The rule that triggered the transition to Q_ACCEPT is the penultimate step in trace
        applied_rule = trace[-2] if len(trace) >= 2 else trace[-1]
        print(f"{noun:<18} | {plural:<20} | {applied_rule}")

    print("-" * 75)
    print("\nDetailed FSM State Traces for Selected Examples:")
    for sample in ["child", "church", "city", "knife", "potato", "cat"]:
        plural, trace = fsm.parse_plural(sample)
        print(f"\nProcessing '{sample}' -> '{plural}':")
        for step in trace:
            print(f"   -> {step}")


if __name__ == "__main__":
    main()
