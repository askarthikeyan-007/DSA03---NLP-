"""
================================================================================
EXPERIMENT 07
Title: Part-of-Speech (POS) Tagging using NLTK
Objective: Implement a Python program using NLTK to perform Part-of-Speech
           tagging on text, map Penn Treebank tags to Universal tags, and
           display linguistic tag descriptions and frequency statistics.
================================================================================
"""

from collections import Counter
import nltk

# Ensure tokenizers and POS models are ready
try:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    nltk.download("universal_tagset", quiet=True)
except Exception:
    pass

# Detailed descriptions of common Penn Treebank POS tags
PENN_TAG_DESCRIPTIONS = {
    "CC": "Coordinating conjunction (e.g. and, but)",
    "CD": "Cardinal number (e.g. one, two, 2026)",
    "DT": "Determiner (e.g. the, a, these)",
    "EX": "Existential there (e.g. there is)",
    "IN": "Preposition / subordinating conjunction (e.g. in, of, on)",
    "JJ": "Adjective (e.g. big, green)",
    "JJR": "Adjective, comparative (e.g. bigger)",
    "JJS": "Adjective, superlative (e.g. biggest)",
    "MD": "Modal (e.g. can, could, will)",
    "NN": "Noun, singular or mass (e.g. dog, intelligence)",
    "NNS": "Noun, plural (e.g. dogs, computers)",
    "NNP": "Proper noun, singular (e.g. Alice, London)",
    "NNPS": "Proper noun, plural (e.g. Americans)",
    "POS": "Possessive ending (e.g. 's)",
    "PRP": "Personal pronoun (e.g. I, he, they)",
    "PRP$": "Possessive pronoun (e.g. my, his, their)",
    "RB": "Adverb (e.g. quickly, very)",
    "RBR": "Adverb, comparative (e.g. faster)",
    "RBS": "Adverb, superlative (e.g. fastest)",
    "TO": "to (to go, to him)",
    "VB": "Verb, base form (e.g. eat, run)",
    "VBD": "Verb, past tense (e.g. ate, ran)",
    "VBG": "Verb, gerund / present participle (e.g. eating, running)",
    "VBN": "Verb, past participle (e.g. eaten, run)",
    "VBP": "Verb, non-3rd person singular present (e.g. write)",
    "VBZ": "Verb, 3rd person singular present (e.g. writes)",
    "WDT": "Wh-determiner (e.g. which, that)",
    "WP": "Wh-pronoun (e.g. who, what)",
    "WRB": "Wh-adverb (e.g. where, when)",
    ".": "Sentence-final punctuation (. ! ?)",
    ",": "Comma",
    ":": "Colon or ellipsis",
}


def main():
    print("=" * 80)
    print("EXPERIMENT 07: PART-OF-SPEECH (POS) TAGGING USING NLTK")
    print("=" * 80)

    sample_text = (
        "Google Antigravity develops cutting-edge artificial intelligence systems. "
        "The enthusiastic engineers quickly solved 15 challenging natural language problems."
    )

    print("Input Text:\n", sample_text)
    print("-" * 80)

    # 1. Tokenization
    tokens = nltk.word_tokenize(sample_text)

    # 2. POS Tagging with Penn Treebank Tagset
    penn_tagged = nltk.pos_tag(tokens)

    # 3. POS Tagging with Universal Tagset
    univ_tagged = nltk.pos_tag(tokens, tagset="universal")

    # Display Tagged Tokens Table
    print(f"{'Token':<18} | {'Penn POS':<10} | {'Universal':<12} | {'Description'}")
    print("-" * 80)
    for (w, p_tag), (_, u_tag) in zip(penn_tagged, univ_tagged):
        desc = PENN_TAG_DESCRIPTIONS.get(p_tag, "Other / Punctuation")
        print(f"{w:<18} | {p_tag:<10} | {u_tag:<12} | {desc}")
    print("-" * 80)

    # 4. Statistical Summary / Frequency Analysis
    tag_counts = Counter(tag for _, tag in penn_tagged)
    univ_counts = Counter(tag for _, tag in univ_tagged)

    print("\n[A] Penn Treebank Tag Distribution:")
    for tag, count in tag_counts.most_common():
        pct = (count / len(tokens)) * 100
        desc = PENN_TAG_DESCRIPTIONS.get(tag, "")
        print(f"   {tag:<6} : {count:>2} ({pct:>5.1f}%) -> {desc}")

    print("\n[B] Universal Tag Distribution:")
    for tag, count in univ_counts.most_common():
        pct = (count / len(tokens)) * 100
        print(f"   {tag:<8} : {count:>2} ({pct:>5.1f}%)")


if __name__ == "__main__":
    main()
