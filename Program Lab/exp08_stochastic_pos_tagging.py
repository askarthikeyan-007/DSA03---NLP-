"""
================================================================================
EXPERIMENT 08
Title: Stochastic Part-of-Speech Tagging using Hidden Markov Model (HMM) & Viterbi
Objective: Implement a probabilistic (stochastic) POS tagger in Python based on
           HMM Bigram transitions, Word emissions, and the Viterbi decoding algorithm.
================================================================================
"""

from collections import defaultdict, Counter
import math


class HMMPOSTagger:
    """
    Hidden Markov Model (HMM) Part-of-Speech Tagger using the Viterbi Algorithm.

    Model Components:
      - States (S): Set of POS tags
      - Observations (V): Words in vocabulary
      - Transition Probabilities A[t_prev][t_curr] = P(t_curr | t_prev)
      - Emission Probabilities B[tag][word] = P(word | tag)
      - Initial Probabilities Pi[tag] = P(tag | START)
    """

    def __init__(self):
        self.tags = set()
        self.vocab = set()
        self.transition_counts = defaultdict(Counter)
        self.emission_counts = defaultdict(Counter)
        self.tag_counts = Counter()
        self.start_counts = Counter()
        self.total_sentences = 0

    def train(self, tagged_sentences):
        """Trains the HMM on a list of tagged sentences [(word, tag), ...]."""
        self.total_sentences = len(tagged_sentences)

        for sentence in tagged_sentences:
            if not sentence:
                continue

            # First word tag for start probability
            first_tag = sentence[0][1]
            self.start_counts[first_tag] += 1

            prev_tag = "<START>"
            for word, tag in sentence:
                word_lower = word.lower()
                self.tags.add(tag)
                self.vocab.add(word_lower)

                self.tag_counts[tag] += 1
                self.emission_counts[tag][word_lower] += 1

                if prev_tag != "<START>":
                    self.transition_counts[prev_tag][tag] += 1
                prev_tag = tag

    def transition_prob(self, prev_tag: str, current_tag: str, alpha: float = 0.01) -> float:
        """Computes smoothed transition probability P(current_tag | prev_tag)."""
        denom = self.tag_counts[prev_tag] + alpha * len(self.tags)
        numer = self.transition_counts[prev_tag][current_tag] + alpha
        return numer / denom

    def emission_prob(self, tag: str, word: str, alpha: float = 0.001) -> float:
        """Computes smoothed emission probability P(word | tag)."""
        w_lower = word.lower()
        denom = self.tag_counts[tag] + alpha * (len(self.vocab) + 1)
        numer = self.emission_counts[tag][w_lower] + (alpha if w_lower in self.vocab else alpha / 10)
        return numer / denom

    def start_prob(self, tag: str, alpha: float = 0.01) -> float:
        """Computes smoothed initial tag probability P(tag | START)."""
        return (self.start_counts[tag] + alpha) / (self.total_sentences + alpha * len(self.tags))

    def viterbi_tag(self, words):
        """
        Applies the Viterbi dynamic programming algorithm to find the most
        probable tag sequence for the given list of words.
        """
        if not words:
            return []

        tags_list = sorted(list(self.tags))
        T = len(words)

        # viterbi_table[t][tag] = log probability of best path up to word t ending in tag
        viterbi_table = [{} for _ in range(T)]
        # backpointer[t][tag] = best previous tag that led to (t, tag)
        backpointer = [{} for _ in range(T)]

        # 1. Initialization (t = 0)
        first_word = words[0]
        for tag in tags_list:
            p_init = self.start_prob(tag)
            p_emit = self.emission_prob(tag, first_word)
            viterbi_table[0][tag] = math.log(p_init) + math.log(p_emit)
            backpointer[0][tag] = None

        # 2. Recursion step (t = 1 to T - 1)
        for t in range(1, T):
            word = words[t]
            for curr_tag in tags_list:
                p_emit = self.emission_prob(curr_tag, word)
                best_prob = -float("inf")
                best_prev = None

                for prev_tag in tags_list:
                    p_trans = self.transition_prob(prev_tag, curr_tag)
                    prob = viterbi_table[t - 1][prev_tag] + math.log(p_trans) + math.log(p_emit)
                    if prob > best_prob:
                        best_prob = prob
                        best_prev = prev_tag

                viterbi_table[t][curr_tag] = best_prob
                backpointer[t][curr_tag] = best_prev

        # 3. Termination
        best_last_prob = -float("inf")
        best_last_tag = None
        for tag in tags_list:
            if viterbi_table[T - 1][tag] > best_last_prob:
                best_last_prob = viterbi_table[T - 1][tag]
                best_last_tag = tag

        # 4. Backtracking path
        best_path = [best_last_tag]
        for t in range(T - 1, 0, -1):
            prev_tag = backpointer[t][best_path[-1]]
            best_path.append(prev_tag)

        best_path.reverse()
        return list(zip(words, best_path))


def main():
    print("=" * 80)
    print("EXPERIMENT 08: STOCHASTIC POS TAGGING (HMM & VITERBI ALGORITHM)")
    print("=" * 80)

    # Annotated training corpus
    training_data = [
        [("The", "DET"), ("dog", "NOUN"), ("barks", "VERB"), ("loudly", "ADV"), (".", ".")],
        [("The", "DET"), ("cat", "NOUN"), ("sat", "VERB"), ("on", "PREP"), ("the", "DET"), ("mat", "NOUN"), (".", ".")],
        [("A", "DET"), ("young", "ADJ"), ("boy", "NOUN"), ("can", "MODAL"), ("fish", "VERB"), (".", ".")],
        [("They", "PRON"), ("caught", "VERB"), ("a", "DET"), ("large", "ADJ"), ("fish", "NOUN"), (".", ".")],
        [("She", "PRON"), ("will", "MODAL"), ("run", "VERB"), ("fast", "ADV"), (".", ".")],
        [("The", "DET"), ("run", "NOUN"), ("was", "VERB"), ("long", "ADJ"), (".", ".")],
        [("He", "PRON"), ("can", "MODAL"), ("can", "VERB"), ("the", "DET"), ("can", "NOUN"), (".", ".")],
        [("Good", "ADJ"), ("students", "NOUN"), ("study", "VERB"), ("hard", "ADV"), (".", ".")],
        [("Natural", "ADJ"), ("language", "NOUN"), ("processing", "NOUN"), ("is", "VERB"), ("exciting", "ADJ"), (".", ".")]
    ]

    tagger = HMMPOSTagger()
    tagger.train(training_data)

    print(f"HMM Model trained on {tagger.total_sentences} sentences.")
    print(f"Unique POS Tags ({len(tagger.tags)}): {sorted(list(tagger.tags))}")
    print(f"Unique Words ({len(tagger.vocab)}): {len(tagger.vocab)} entries.")

    # Demonstrate ambiguity resolution (e.g. 'can' as MODAL vs VERB vs NOUN, 'fish' as NOUN vs VERB)
    test_sentences = [
        ["The", "young", "boy", "can", "fish", "."],
        ["They", "caught", "a", "fish", "."],
        ["He", "can", "run", "fast", "."],
        ["The", "run", "was", "long", "."],
        ["She", "will", "study", "hard", "."]
    ]

    print("\n--- Stochastic Tagging via Viterbi Decoding ---")
    for words in test_sentences:
        tagged = tagger.viterbi_tag(words)
        tagged_str = " ".join([f"{w}/{t}" for w, t in tagged])
        print(f"\nSentence: {' '.join(words)}")
        print(f"Tagged  : {tagged_str}")

    print("\n--- Key Transition Probabilities ---")
    transitions_to_show = [("DET", "NOUN"), ("DET", "ADJ"), ("MODAL", "VERB"), ("NOUN", "VERB"), ("ADJ", "NOUN")]
    for p_tag, c_tag in transitions_to_show:
        prob = tagger.transition_prob(p_tag, c_tag)
        print(f"   P({c_tag:<6} | {p_tag:<6}) = {prob:.4f}")

    print("\n--- Disambiguating Ambiguous Words ---")
    for ambig_word in ["can", "fish", "run"]:
        print(f"\nWord: '{ambig_word}'")
        for tag in ["NOUN", "VERB", "MODAL"]:
            if tag in tagger.tags:
                prob = tagger.emission_prob(tag, ambig_word)
                print(f"   P('{ambig_word}' | {tag:<6}) = {prob:.4f}")


if __name__ == "__main__":
    main()
