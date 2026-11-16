"""
================================================================================
EXPERIMENT 22
Title: Reference Resolution (Anaphora Resolution) in Discourse
Objective: Implement a Python program that performs reference resolution to
           resolve pronominal anaphora by matching pronouns to their correct
           antecedents using gender, number, animacy, and recency constraints.
================================================================================
"""

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EntityCandidate:
    text: str
    sentence_idx: int
    gender: str       # 'M', 'F', 'N' (neuter)
    number: str       # 'SG', 'PL'
    animacy: str      # 'HUMAN', 'NON_HUMAN'
    is_subject: bool


class RuleBasedReferenceResolver:
    """
    Pronominal Reference Resolver using Salience and Agreement Constraints.
    """

    PRONOUN_MAP = {
        "he":   {"gender": "M", "number": "SG", "animacy": "HUMAN"},
        "him":  {"gender": "M", "number": "SG", "animacy": "HUMAN"},
        "his":  {"gender": "M", "number": "SG", "animacy": "HUMAN"},
        "she":  {"gender": "F", "number": "SG", "animacy": "HUMAN"},
        "her":  {"gender": "F", "number": "SG", "animacy": "HUMAN"},
        "hers": {"gender": "F", "number": "SG", "animacy": "HUMAN"},
        "it":   {"gender": "N", "number": "SG", "animacy": "NON_HUMAN"},
        "its":  {"gender": "N", "number": "SG", "animacy": "NON_HUMAN"},
        "they": {"gender": "ANY", "number": "PL", "animacy": "ANY"},
        "them": {"gender": "ANY", "number": "PL", "animacy": "ANY"},
        "their":{"gender": "ANY", "number": "PL", "animacy": "ANY"},
    }

    KNOWN_ENTITIES = {
        # Humans - Male
        "john": ("M", "SG", "HUMAN"),
        "david": ("M", "SG", "HUMAN"),
        "bob": ("M", "SG", "HUMAN"),
        "the boy": ("M", "SG", "HUMAN"),
        "the man": ("M", "SG", "HUMAN"),
        "the king": ("M", "SG", "HUMAN"),
        "the father": ("M", "SG", "HUMAN"),

        # Humans - Female
        "mary": ("F", "SG", "HUMAN"),
        "alice": ("F", "SG", "HUMAN"),
        "the girl": ("F", "SG", "HUMAN"),
        "the woman": ("F", "SG", "HUMAN"),
        "the queen": ("F", "SG", "HUMAN"),
        "the mother": ("F", "SG", "HUMAN"),

        # Non-Humans / Inanimate (Singular)
        "the dog": ("N", "SG", "NON_HUMAN"),
        "the cat": ("N", "SG", "NON_HUMAN"),
        "the book": ("N", "SG", "NON_HUMAN"),
        "the car": ("N", "SG", "NON_HUMAN"),
        "the robot": ("N", "SG", "NON_HUMAN"),
        "the computer": ("N", "SG", "NON_HUMAN"),

        # Plural Entities
        "the dogs": ("ANY", "PL", "NON_HUMAN"),
        "the students": ("ANY", "PL", "HUMAN"),
        "the programmers": ("ANY", "PL", "HUMAN"),
        "the books": ("ANY", "PL", "NON_HUMAN"),
    }

    def resolve_discourse(self, text: str):
        """Resolves anaphoric pronouns across consecutive discourse sentences."""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

        discourse_entities: List[EntityCandidate] = []
        resolutions = []

        resolved_sentences = []

        for sent_idx, sentence in enumerate(sentences):
            words = sentence.rstrip(".!?").split()
            current_sent_resolved = list(words)

            # Step 1: Discover candidate entities in the sentence
            for name, (g, num, anim) in self.KNOWN_ENTITIES.items():
                pattern = r"\b" + re.escape(name) + r"\b"
                if re.search(pattern, sentence, re.IGNORECASE):
                    is_subj = sentence.lower().startswith(name.lower())
                    discourse_entities.append(EntityCandidate(
                        text=name.title(),
                        sentence_idx=sent_idx,
                        gender=g,
                        number=num,
                        animacy=anim,
                        is_subject=is_subj
                    ))

            # Step 2: Identify and resolve pronouns
            for token_idx, word in enumerate(words):
                w_clean = re.sub(r"[^\w]", "", word).lower()
                if w_clean in self.PRONOUN_MAP:
                    p_info = self.PRONOUN_MAP[w_clean]
                    # Score candidate antecedents
                    best_cand = None
                    best_score = -100

                    # Search previous entities
                    for cand in reversed(discourse_entities):
                        if cand.sentence_idx > sent_idx:
                            continue

                        # Check Agreement Constraints
                        if p_info["gender"] != "ANY" and cand.gender != "ANY" and p_info["gender"] != cand.gender:
                            continue
                        if p_info["number"] != "ANY" and cand.number != "ANY" and p_info["number"] != cand.number:
                            continue
                        if p_info["animacy"] != "ANY" and cand.animacy != "ANY" and p_info["animacy"] != cand.animacy:
                            continue

                        # Salience Scoring
                        recency = 10 - (sent_idx - cand.sentence_idx) * 3
                        subject_bonus = 5 if cand.is_subject else 0
                        score = recency + subject_bonus

                        if score > best_score:
                            best_score = score
                            best_cand = cand

                    if best_cand:
                        resolutions.append((sent_idx + 1, word, best_cand.text, best_score))
                        current_sent_resolved[token_idx] = f"[{best_cand.text}]"

            resolved_sentences.append(" ".join(current_sent_resolved) + ".")

        return resolutions, " ".join(resolved_sentences)


def main():
    print("=" * 80)
    print("EXPERIMENT 22: REFERENCE RESOLUTION (ANAPHORA RESOLUTION)")
    print("=" * 80)

    discourse_text = (
        "John bought a new computer yesterday. He tested it thoroughly. "
        "Mary met Alice in the library. She gave her a book. "
        "The robot charged on the dock. It operated smoothly."
    )

    print("Input Discourse Text:\n", discourse_text)
    print("-" * 80)

    resolver = RuleBasedReferenceResolver()
    resolutions, resolved_text = resolver.resolve_discourse(discourse_text)

    print("\nIdentified Pronoun Reference Resolutions:")
    print("-" * 80)
    print(f"{'Sentence #':<12} | {'Pronoun':<10} | {'Resolved Antecedent':<22} | {'Salience Score'}")
    print("-" * 80)
    for sent_no, pronoun, antecedent, score in resolutions:
        print(f"Sentence {sent_no:<3} | {pronoun:<10} | {antecedent:<22} | {score}")
    print("-" * 80)

    print("\nResolved Discourse Text (with Antecedents Substituted):")
    print(" ", resolved_text)


if __name__ == "__main__":
    main()
