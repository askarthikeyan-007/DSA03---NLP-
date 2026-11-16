"""
================================================================================
EXPERIMENT 06
Title: N-gram Language Model for Text Generation (Bigram Model)
Objective: Implement a Bigram language model in Python to calculate conditional
           probabilities and generate text from training corpora.
================================================================================
"""

import random
import re
from collections import defaultdict, Counter


class BigramLanguageModel:
    """
    Bigram Language Model for text generation based on maximum likelihood estimation (MLE):
        P(w_i | w_{i-1}) = Count(w_{i-1}, w_i) / Count(w_{i-1})
    """

    def __init__(self):
        # unigram_counts[w1] = count
        self.unigram_counts = Counter()
        # bigram_counts[w1][w2] = count
        self.bigram_counts = defaultdict(Counter)
        # vocabulary
        self.vocab = set()

    def train(self, corpus: str):
        """Tokenizes corpus, adds sentence boundaries, and computes counts."""
        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", corpus.strip())

        for sent in sentences:
            tokens = self._tokenize(sent)
            if not tokens:
                continue

            # Add start and end of sentence markers
            padded_tokens = ["<s>"] + tokens + ["</s>"]

            for i in range(len(padded_tokens) - 1):
                w1 = padded_tokens[i]
                w2 = padded_tokens[i + 1]

                self.unigram_counts[w1] += 1
                self.bigram_counts[w1][w2] += 1
                self.vocab.add(w1)
                self.vocab.add(w2)

    def _tokenize(self, text: str):
        """Cleans and extracts words and punctuation tokens."""
        text = text.lower()
        tokens = re.findall(r"\b[a-z0-9'-]+\b|[.!?,]", text)
        return tokens

    def probability(self, w1: str, w2: str) -> float:
        """Returns conditional probability P(w2 | w1)."""
        w1_count = self.unigram_counts.get(w1, 0)
        if w1_count == 0:
            return 0.0
        return self.bigram_counts[w1][w2] / w1_count

    def generate(self, seed_word: str = "<s>", max_length: int = 25, mode: str = "probabilistic", seed: int = 42) -> str:
        """
        Generates text beginning with seed_word.
        mode: 'probabilistic' (weighted random) or 'greedy' (argmax)
        """
        if seed is not None:
            random.seed(seed)

        current_word = seed_word.lower()
        if current_word not in self.bigram_counts and current_word != "<s>":
            return f"Seed word '{seed_word}' not found in vocabulary."

        output = []
        if current_word != "<s>":
            output.append(current_word)

        for _ in range(max_length):
            transitions = self.bigram_counts.get(current_word)
            if not transitions:
                break

            words = list(transitions.keys())
            counts = list(transitions.values())

            if mode == "greedy":
                # Choose token with highest count/probability
                next_word = words[counts.index(max(counts))]
            else:
                # Weighted random choice based on bigram frequency
                next_word = random.choices(words, weights=counts, k=1)[0]

            if next_word == "</s>":
                break

            output.append(next_word)
            current_word = next_word

        # Format output text nicely
        result = " ".join(output)
        result = re.sub(r'\s+([.!?,])', r'\1', result)
        return result.capitalize()

    def print_transitions(self, target_words):
        """Prints conditional probability distributions for specific words."""
        print("\n--- Bigram Conditional Probabilities P(W_next | W_current) ---")
        for word in target_words:
            if word not in self.bigram_counts:
                continue
            total = self.unigram_counts[word]
            print(f"\nConditioned on '{word}' (Total occurrences: {total}):")
            for next_w, count in self.bigram_counts[word].most_common(5):
                prob = count / total
                bar = "#" * int(prob * 20)
                print(f"   P({next_w:<12} | {word:<10}) = {prob:.4f}  {bar}")


def main():
    print("=" * 75)
    print("EXPERIMENT 06: BIGRAM LANGUAGE MODEL FOR TEXT GENERATION")
    print("=" * 75)

    training_corpus = """
    Natural language processing is a subfield of artificial intelligence.
    Artificial intelligence is transforming modern science and technology.
    Natural language processing helps computers understand human language.
    Human language is rich and complex. Computers process language using statistical models.
    Statistical models and machine learning algorithms learn patterns from large corpora.
    Artificial intelligence algorithms make predictions and generate text accurately.
    Natural language processing systems analyze text, speech, and dialog.
    Machine learning enables computers to learn from data without explicit programming.
    Computers can understand text and generate text with high fluency.
    """

    model = BigramLanguageModel()
    model.train(training_corpus)

    print(f"Corpus processed.")
    print(f"Total Unique Vocabulary: {len(model.vocab)}")
    print(f"Total Unigrams tracked: {sum(model.unigram_counts.values())}")

    # Inspect transition probabilities
    model.print_transitions(["natural", "artificial", "language", "computers", "is"])

    # Generate sentences
    print("\n" + "=" * 75)
    print("TEXT GENERATION DEMONSTRATION")
    print("=" * 75)

    print("\n[1] Starting with <s> (Sentence Start), Probabilistic Generation:")
    for i in range(1, 4):
        gen_text = model.generate(seed_word="<s>", max_length=20, mode="probabilistic", seed=i * 10)
        print(f"   Sentence #{i}: \"{gen_text}\"")

    print("\n[2] Starting with specific seed words:")
    seeds = ["natural", "artificial", "computers", "machine"]
    for seed in seeds:
        gen_text = model.generate(seed_word=seed, max_length=15, mode="probabilistic", seed=100)
        print(f"   Seed '{seed:<11}': \"{gen_text}\"")

    print("\n[3] Greedy (Argmax) Generation:")
    greedy_text = model.generate(seed_word="natural", max_length=15, mode="greedy")
    print(f"   Greedy from 'natural': \"{greedy_text}\"")


if __name__ == "__main__":
    main()
