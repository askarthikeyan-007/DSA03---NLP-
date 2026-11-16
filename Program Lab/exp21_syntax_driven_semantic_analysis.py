"""
================================================================================
EXPERIMENT 21
Title: Syntax-Driven Semantic Analysis (Noun Phrase Meaning Extraction)
Objective: Implement a Python program that performs syntax-driven semantic
           analysis by parsing sentences to extract Noun Phrases (NP) and
           linking their head nouns to lexical semantics (WordNet).
================================================================================
"""

import nltk
from nltk import RegexpParser
from nltk.corpus import wordnet as wn

# Ensure required resources are available
try:
    nltk.download("punkt", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
except Exception:
    pass


class SyntaxDrivenSemanticAnalyzer:
    """
    Performs syntax-driven semantic analysis by:
      1. Syntactic Chunking: Identifying Noun Phrases via grammar patterns.
      2. Constituent Decomposition: Extracting Determiners, Modifiers, and Head Nouns.
      3. Semantic Interpretation: Grounding the head noun in WordNet ontology.
    """

    def __init__(self):
        # Chunk Grammar for Noun Phrases
        # NP matches optional Determiner, zero or more Adjectives, and one or more Nouns
        chunk_grammar = r"""
            NP: {<DT|PRP\$>?<JJ.*>*<NN.*>+}
        """
        self.chunk_parser = RegexpParser(chunk_grammar)

    def extract_noun_phrases(self, sentence: str):
        """Tokenizes, POS tags, and parses text into NP chunks."""
        tokens = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(tokens)
        parse_tree = self.chunk_parser.parse(tagged)

        noun_phrases = []
        for subtree in parse_tree.subtrees(lambda t: t.label() == "NP"):
            np_words = [word for word, tag in subtree.leaves()]
            np_tags = [tag for word, tag in subtree.leaves()]
            noun_phrases.append((np_words, np_tags))

        return noun_phrases, parse_tree

    def analyze_np_semantics(self, np_words, np_tags):
        """Analyzes internal composition and WordNet meaning for a Noun Phrase."""
        head_noun = np_words[-1]  # The head noun in English NPs is typically the final noun
        modifiers = np_words[:-1]

        # Retrieve WordNet semantics for head noun
        synsets = wn.synsets(head_noun, pos=wn.NOUN)
        if synsets:
            primary_synset = synsets[0]
            definition = primary_synset.definition()
            hypernyms = [h.name().split(".")[0] for h in primary_synset.hypernyms()]
            semantic_category = hypernyms[0] if hypernyms else "entity"
        else:
            definition = "Specific named entity or domain term."
            semantic_category = "entity"

        return {
            "Full NP": " ".join(np_words),
            "Head Noun": head_noun,
            "Modifiers": modifiers if modifiers else ["None"],
            "Semantic Category (Hypernym)": semantic_category,
            "Lexical Definition": definition
        }


def main():
    print("=" * 80)
    print("EXPERIMENT 21: SYNTAX-DRIVEN SEMANTIC ANALYSIS")
    print("=" * 80)

    analyzer = SyntaxDrivenSemanticAnalyzer()

    sentences = [
        "The intelligent autonomous robot navigated through the dark forest.",
        "A brilliant scientist discovered an innovative mathematical algorithm."
    ]

    for sent_idx, sentence in enumerate(sentences, 1):
        print(f"\n[Sentence #{sent_idx}]: \"{sentence}\"")
        print("-" * 80)

        nps, tree = analyzer.extract_noun_phrases(sentence)
        print(f"Extracted {len(nps)} Noun Phrase(s):")

        for np_idx, (np_words, np_tags) in enumerate(nps, 1):
            sem = analyzer.analyze_np_semantics(np_words, np_tags)
            print(f"\n  (NP #{np_idx}): \"{sem['Full NP']}\"")
            print(f"     * Head Noun            : '{sem['Head Noun']}'")
            print(f"     * Modifiers / Specifier : {sem['Modifiers']}")
            print(f"     * Semantic Class (Type): [{sem['Semantic Category (Hypernym)']}]")
            print(f"     * Meaning (Definition) : {sem['Lexical Definition']}")

        print("\nSyntactic Chunk Tree Structure:")
        tree.pretty_print()


if __name__ == "__main__":
    main()
