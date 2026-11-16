"""
================================================================================
EXPERIMENT 20
Title: Information Retrieval System using TF-IDF for Document Ranking
Objective: Implement a complete Information Retrieval (IR) system in Python
           using Term Frequency - Inverse Document Frequency (TF-IDF) and
           Cosine Similarity to rank documents in response to user queries.
================================================================================
"""

import math
import re
from collections import Counter


class TFIDFInformationRetrieval:
    """
    TF-IDF Information Retrieval engine from first principles.

    Formulas:
      - TF(t, d) = count(t, d) / total_words_in_d
      - IDF(t)   = ln((N + 1) / (DF(t) + 1)) + 1
      - TF-IDF   = TF(t, d) * IDF(t)
      - Similarity(q, d) = (q . d) / (||q|| * ||d||)
    """

    def __init__(self, documents: dict):
        self.doc_ids = list(documents.keys())
        self.docs = documents
        self.N = len(documents)

        # Preprocessing: tokenize and clean documents
        self.doc_tokens = {doc_id: self._tokenize(text) for doc_id, text in documents.items()}

        # Build vocabulary and Document Frequencies (DF)
        self.vocab = set()
        self.df = Counter()
        for tokens in self.doc_tokens.values():
            unique_terms = set(tokens)
            self.vocab.update(unique_terms)
            for term in unique_terms:
                self.df[term] += 1

        # Compute IDF weights
        self.idf = {}
        for term in self.vocab:
            self.idf[term] = math.log((self.N + 1) / (self.df[term] + 1)) + 1.0

        # Precompute document TF-IDF vectors
        self.doc_vectors = {}
        for doc_id, tokens in self.doc_tokens.items():
            self.doc_vectors[doc_id] = self._compute_tfidf_vector(tokens)

    def _tokenize(self, text: str):
        """Converts text to normalized lowercase word tokens."""
        return re.findall(r"\b[a-z0-9]+\b", text.lower())

    def _compute_tfidf_vector(self, tokens):
        """Computes a sparse TF-IDF dictionary vector for a token list."""
        tf_counts = Counter(tokens)
        total_tokens = len(tokens)
        vector = {}

        for term, count in tf_counts.items():
            tf = count / total_tokens if total_tokens > 0 else 0
            idf_val = self.idf.get(term, 0.0)
            vector[term] = tf * idf_val

        return vector

    def _vector_norm(self, vector: dict) -> float:
        """Euclidean L2 norm of a vector."""
        return math.sqrt(sum(v ** 2 for v in vector.values()))

    def _cosine_similarity(self, vec1: dict, vec2: dict) -> float:
        """Computes Cosine Similarity between two sparse vectors."""
        norm1 = self._vector_norm(vec1)
        norm2 = self._vector_norm(vec2)
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0

        # Dot product
        dot = sum(vec1[term] * vec2[term] for term in vec1 if term in vec2)
        return dot / (norm1 * norm2)

    def search(self, query: str, top_k: int = 3):
        """Ranks documents against query using cosine similarity."""
        q_tokens = self._tokenize(query)
        q_vector = self._compute_tfidf_vector(q_tokens)

        scores = []
        for doc_id, d_vector in self.doc_vectors.items():
            sim = self._cosine_similarity(q_vector, d_vector)
            scores.append((doc_id, sim, self.docs[doc_id]))

        # Sort descending by similarity score
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def main():
    print("=" * 80)
    print("EXPERIMENT 20: INFORMATION RETRIEVAL USING TF-IDF RANKING")
    print("=" * 80)

    # Document Corpus
    corpus = {
        "DOC_01 (NLP)": "Natural language processing involves computational linguistics and machine learning models to analyze human text.",
        "DOC_02 (AI & Robots)": "Artificial intelligence and autonomous robotics allow intelligent machines to solve complex problems.",
        "DOC_03 (Climate)": "Global climate change increases average surface temperatures, causing severe weather patterns and glacier melting.",
        "DOC_04 (Astronomy)": "Astronomers use optical and radio telescopes to observe distant stars, black holes, and galaxies in space.",
        "DOC_05 (Search Engines)": "Information retrieval systems and web search engines index documents to match user query terms and rank relevant results."
    }

    ir_engine = TFIDFInformationRetrieval(corpus)

    print(f"Total Documents in Index: {ir_engine.N}")
    print(f"Total Vocabulary Size   : {len(ir_engine.vocab)} terms\n")

    # Sample Queries to test relevance
    test_queries = [
        "computational natural language models",
        "telescopes observing stars and galaxies",
        "search engines retrieving indexed documents",
        "renewable climate temperature change"
    ]

    for q_idx, query in enumerate(test_queries, 1):
        print("=" * 80)
        print(f"QUERY #{q_idx}: \"{query}\"")
        print("=" * 80)

        results = ir_engine.search(query, top_k=3)

        print(f"{'Rank':<5} | {'Document ID':<24} | {'Cosine Sim':<12} | {'Document Excerpt'}")
        print("-" * 80)
        for rank, (doc_id, score, text) in enumerate(results, 1):
            excerpt = text[:60] + "..." if len(text) > 60 else text
            print(f"{rank:<5} | {doc_id:<24} | {score:<12.4f} | {excerpt}")
        print("-" * 80)


if __name__ == "__main__":
    main()
