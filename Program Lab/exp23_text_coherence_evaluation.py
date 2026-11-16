"""
================================================================================
EXPERIMENT 23
Title: Text Coherence Evaluation in Discourse
Objective: Develop a Python program to evaluate the coherence of texts using
           lexical cohesion, adjacent sentence semantic overlap, and discourse
           connective analysis.
================================================================================
"""

import math
import re
from collections import Counter


class TextCoherenceEvaluator:
    """
    Evaluates the coherence of a discourse based on:
      1. Lexical Overlap between adjacent sentences (Jaccard similarity).
      2. Semantic vector continuity (Cosine similarity using TF-IDF).
      3. Discourse Connectives density (transitional and logical markers).
      4. Composite Coherence Index (0.0 - 1.0).
    """

    DISCOURSE_CONNECTIVES = {
        "addition": ["furthermore", "moreover", "in addition", "additionally", "also"],
        "contrast": ["however", "nevertheless", "on the other hand", "yet", "although", "despite"],
        "causality": ["therefore", "consequently", "thus", "because", "as a result", "hence"],
        "temporal": ["subsequently", "meanwhile", "afterwards", "finally", "then"]
    }

    STOPWORDS = {
        "the", "a", "an", "in", "on", "of", "and", "or", "is", "was", "were", "are",
        "to", "for", "with", "at", "by", "from", "it", "this", "that"
    }

    def _tokenize(self, text: str):
        words = re.findall(r"\b[a-z]{3,}\b", text.lower())
        return [w for w in words if w not in self.STOPWORDS]

    def _split_sentences(self, text: str):
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def evaluate(self, text: str):
        sentences = self._split_sentences(text)
        if len(sentences) < 2:
            return {"error": "Need at least 2 sentences to evaluate coherence."}

        sent_tokens = [self._tokenize(s) for s in sentences]

        # 1. Adjacent Sentence Lexical Overlap (Jaccard Similarity)
        jaccard_scores = []
        for i in range(len(sent_tokens) - 1):
            s1 = set(sent_tokens[i])
            s2 = set(sent_tokens[i + 1])
            union = s1.union(s2)
            if union:
                jaccard = len(s1.intersection(s2)) / len(union)
            else:
                jaccard = 0.0
            jaccard_scores.append(jaccard)

        avg_jaccard = sum(jaccard_scores) / len(jaccard_scores)

        # 2. Vector Cosine Similarity between adjacent sentences
        all_vocab = set()
        for tok_list in sent_tokens:
            all_vocab.update(tok_list)

        def make_vec(tok_list):
            counts = Counter(tok_list)
            return [counts.get(w, 0) for w in all_vocab]

        cosine_scores = []
        for i in range(len(sent_tokens) - 1):
            v1 = make_vec(sent_tokens[i])
            v2 = make_vec(sent_tokens[i + 1])
            dot = sum(a * b for a, b in zip(v1, v2))
            norm1 = math.sqrt(sum(a ** 2 for a in v1))
            norm2 = math.sqrt(sum(b ** 2 for b in v2))
            cos = dot / (norm1 * norm2) if (norm1 * norm2) > 0 else 0.0
            cosine_scores.append(cos)

        avg_cosine = sum(cosine_scores) / len(cosine_scores)

        # 3. Discourse Connectives Analysis
        lower_text = text.lower()
        connectives_found = []
        for category, conn_list in self.DISCOURSE_CONNECTIVES.items():
            for conn in conn_list:
                if re.search(r"\b" + re.escape(conn) + r"\b", lower_text):
                    connectives_found.append((conn, category))

        connective_score = min(len(connectives_found) / len(sentences), 1.0)

        # 4. Composite Coherence Index (Weighted Sum)
        composite_index = (0.45 * avg_cosine) + (0.35 * avg_jaccard) + (0.20 * connective_score)

        return {
            "sentence_count": len(sentences),
            "adjacent_jaccard_scores": jaccard_scores,
            "avg_jaccard_overlap": avg_jaccard,
            "adjacent_cosine_scores": cosine_scores,
            "avg_cosine_similarity": avg_cosine,
            "connectives_found": connectives_found,
            "connective_score": connective_score,
            "composite_coherence_index": composite_index,
        }


def main():
    print("=" * 80)
    print("EXPERIMENT 23: TEXT COHERENCE EVALUATION")
    print("=" * 80)

    # Coherent Paragraph (Thematic progression with discourse markers)
    coherent_text = (
        "Artificial intelligence models require extensive training on diverse text corpora. "
        "These models learn intricate syntactic structures and semantic relationships from data. "
        "Consequently, natural language systems can generate accurate and context-aware responses. "
        "Furthermore, modern evaluators verify the logical coherence and factual consistency of the outputs."
    )

    # Incoherent Paragraph (Disjointed, unrelated sentences)
    incoherent_text = (
        "The recipe for chocolate cake requires fresh butter, eggs, and refined sugar. "
        "Quantum mechanics describes the fundamental physical properties of nature at subatomic scales. "
        "The ancient pyramids in Egypt were constructed thousands of years ago. "
        "Yesterday my bicycle tire had a puncture on the highway."
    )

    evaluator = TextCoherenceEvaluator()

    for label, text in [("COHERENT DISCOURSE", coherent_text), ("INCOHERENT DISCOURSE", incoherent_text)]:
        print("\n" + "=" * 80)
        print(f"ANALYZING: {label}")
        print("=" * 80)
        print("Text:\n", text)
        print("-" * 80)

        res = evaluator.evaluate(text)

        print(f"Sentence Count                    : {res['sentence_count']}")
        print(f"Average Adjacent Lexical Jaccard  : {res['avg_jaccard_overlap']:.4f}")
        print(f"Average Adjacent Cosine Sim       : {res['avg_cosine_similarity']:.4f}")
        print(f"Discourse Connectives Identified  : {res['connectives_found']}")
        print(f"Connective Density Score          : {res['connective_score']:.4f}")
        print("-" * 80)
        print(f"COMPOSITE COHERENCE INDEX (0-1)   : {res['composite_coherence_index']:.4f}")

        rating = "HIGH COHERENCE" if res['composite_coherence_index'] > 0.3 else "LOW COHERENCE (DISJOINTED)"
        print(f"Qualitative Assessment            : {rating}")


if __name__ == "__main__":
    main()
