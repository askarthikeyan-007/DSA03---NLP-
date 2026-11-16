"""
================================================================================
EXPERIMENT 03
Title: Morphological Analysis using NLTK
Objective: Demonstrate morphological analysis in Python using NLTK, including
           stemming algorithms, WordNet lemmatization with POS context, and
           morpheme decomposition.
================================================================================
"""

import nltk
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Ensure required NLTK datasets are downloaded
for resource in ["punkt", "punkt_tab", "wordnet", "omw-1.4", "averaged_perceptron_tagger_eng"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass


def get_wordnet_pos(treebank_tag):
    """Maps Penn Treebank POS tags to WordNet POS tags."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def analyze_morphemes(word):
    """Simple morphological decomposition into prefix, root, and suffix."""
    known_prefixes = ["un", "dis", "re", "in", "im", "pre", "mis", "non"]
    known_suffixes = ["ing", "ed", "ly", "es", "s", "ness", "able", "ment", "ful", "less"]

    prefix, suffix = "", ""
    current = word.lower()

    for p in known_prefixes:
        if current.startswith(p) and len(current) > len(p) + 2:
            prefix = p
            current = current[len(p):]
            break

    for s in known_suffixes:
        if current.endswith(s) and len(current) > len(s) + 2:
            suffix = s
            current = current[:-len(s)]
            break

    root = current
    return {"Word": word, "Prefix": prefix or "-", "Root/Stem": root, "Suffix": suffix or "-"}


def main():
    print("=" * 75)
    print("EXPERIMENT 03: MORPHOLOGICAL ANALYSIS USING NLTK")
    print("=" * 75)

    sample_words = [
        "cats", "running", "unhappiness", "better", "studies", "flying",
        "generously", "replaying", "fairness", "wolves", "disagreeable"
    ]

    # 1. Stemming Comparison (Porter vs. Lancaster vs. Snowball)
    print("\n[1] Stemming Comparison Across Multiple Algorithms:")
    porter = PorterStemmer()
    lancaster = LancasterStemmer()
    snowball = SnowballStemmer(language="english")

    print("-" * 75)
    print(f"{'Original Word':<16} | {'Porter':<15} | {'Lancaster':<15} | {'Snowball':<15}")
    print("-" * 75)
    for w in sample_words:
        p_stem = porter.stem(w)
        l_stem = lancaster.stem(w)
        s_stem = snowball.stem(w)
        print(f"{w:<16} | {p_stem:<15} | {l_stem:<15} | {s_stem:<15}")
    print("-" * 75)

    # 2. Lemmatization with WordNet Lemmatizer
    print("\n[2] Lemmatization with WordNet (with POS context):")
    lemmatizer = WordNetLemmatizer()

    test_lemmas = [
        ("cats", wordnet.NOUN),
        ("running", wordnet.VERB),
        ("ran", wordnet.VERB),
        ("better", wordnet.ADJ),
        ("wolves", wordnet.NOUN),
        ("feet", wordnet.NOUN),
        ("corpora", wordnet.NOUN),
        ("was", wordnet.VERB),
    ]

    print("-" * 75)
    print(f"{'Word':<12} | {'POS Tag':<15} | {'Lemmatized Form':<20} | {'Stemmed (Porter)'}")
    print("-" * 75)
    for word, pos in test_lemmas:
        lemma = lemmatizer.lemmatize(word, pos=pos)
        stem = porter.stem(word)
        pos_str = "VERB" if pos == wordnet.VERB else ("NOUN" if pos == wordnet.NOUN else "ADJ")
        print(f"{word:<12} | {pos_str:<15} | {lemma:<20} | {stem}")
    print("-" * 75)

    # 3. Sentence-Level Morphological Analysis with POS Tagging
    sentence = "The active children were happily studying different historical languages."
    print(f"\n[3] Sentence-Level Morphological Breakdown:\nSentence: \"{sentence}\"")

    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)

    print("-" * 75)
    print(f"{'Token':<14} | {'Penn POS':<10} | {'WordNet POS':<12} | {'Lemma':<14} | {'Stem'}")
    print("-" * 75)
    for word, tag in tagged:
        wn_pos = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(word.lower(), pos=wn_pos)
        stem = porter.stem(word.lower())
        print(f"{word:<14} | {tag:<10} | {wn_pos:<12} | {lemma:<14} | {stem}")
    print("-" * 75)

    # 4. Affix / Morpheme Segmentation Analysis
    print("\n[4] Morpheme Segmentation (Prefix + Base + Suffix):")
    complex_words = ["unhappiness", "replaying", "disagreeable", "unfairness", "hopelessness", "previewing"]
    print("-" * 75)
    print(f"{'Word':<16} | {'Prefix':<10} | {'Root / Base':<15} | {'Suffix':<12}")
    print("-" * 75)
    for cw in complex_words:
        res = analyze_morphemes(cw)
        print(f"{res['Word']:<16} | {res['Prefix']:<10} | {res['Root/Stem']:<15} | {res['Suffix']:<12}")
    print("-" * 75)


if __name__ == "__main__":
    main()
