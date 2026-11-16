"""
================================================================================
EXPERIMENT 10
Title: Transformation-Based Tagging (Brill-Style Tagging)
Objective: Implement a Transformation-Based Tagging (TBL) algorithm in Python
           that starts with a baseline tagger and iteratively applies contextual
           transformation rules to correct tagging errors.
================================================================================
"""

from dataclasses import dataclass
from typing import List, Callable


@dataclass
class TransformationRule:
    rule_id: str
    description: str
    from_tag: str
    to_tag: str
    condition: Callable  # function(tokens, tags, index) -> bool


class TransformationBasedTagger:
    """
    Transformation-Based (Brill) Tagger.
    Architecture:
      1. Baseline Tagging: Assigns default initial tag (e.g. NN) or lexical prior.
      2. Rule Application: Iteratively evaluates contextual rules and applies
         transformations to fix erroneous tags.
    """

    def __init__(self, default_tag: str = "NN"):
        self.default_tag = default_tag
        self.rules: List[TransformationRule] = []

    def add_rule(self, rule: TransformationRule):
        self.rules.append(rule)

    def baseline_tag(self, tokens: List[str]) -> List[str]:
        """Simple baseline: Assign default tag, with closed-class dictionary lookup."""
        closed_class = {
            "the": "DT", "a": "DT", "an": "DT", "this": "DT", "that": "DT",
            "can": "MD", "could": "MD", "will": "MD", "would": "MD", "may": "MD",
            "in": "IN", "on": "IN", "at": "IN", "to": "TO", "of": "IN",
            "and": "CC", "or": "CC", "but": "CC",
            "he": "PRP", "she": "PRP", "it": "PRP", "they": "PRP", "i": "PRP",
            ".": ".", ",": ",", "!": ".", "?": "."
        }
        return [closed_class.get(t.lower(), self.default_tag) for t in tokens]

    def tag_with_trace(self, tokens: List[str]):
        """Runs baseline tagging, then applies transformation rules sequentially with trace."""
        current_tags = self.baseline_tag(tokens)
        history = [("BASELINE INITIALIZATION", list(current_tags), "Assigned initial baseline tags")]

        for rule in self.rules:
            changed_indices = []
            new_tags = list(current_tags)

            for i in range(len(tokens)):
                if current_tags[i] == rule.from_tag and rule.condition(tokens, current_tags, i):
                    new_tags[i] = rule.to_tag
                    changed_indices.append((i, tokens[i], rule.from_tag, rule.to_tag))

            if changed_indices:
                current_tags = new_tags
                history.append((rule.rule_id, list(current_tags), changed_indices))

        return current_tags, history


def setup_sample_rules(tagger: TransformationBasedTagger):
    """Configures contextual transformation rules."""

    # Rule 1: Change NN to VB if preceding word has tag MD (e.g. "can fish", "will run")
    def cond_md_before(tokens, tags, i):
        return i > 0 and tags[i - 1] == "MD"

    tagger.add_rule(TransformationRule(
        rule_id="RULE_01",
        description="Change NN -> VB if preceding word is Modal (MD)",
        from_tag="NN",
        to_tag="VB",
        condition=cond_md_before
    ))

    # Rule 2: Change VB to NN if preceding word is Determiner (DT) (e.g. "the can", "the run")
    def cond_dt_before(tokens, tags, i):
        return i > 0 and tags[i - 1] == "DT"

    tagger.add_rule(TransformationRule(
        rule_id="RULE_02",
        description="Change VB -> NN if preceding word is Determiner (DT)",
        from_tag="VB",
        to_tag="NN",
        condition=cond_dt_before
    ))

    # Rule 3: Change NN to VBD if word ends in 'ed' (e.g. "walked", "kicked")
    def cond_ends_ed(tokens, tags, i):
        return tokens[i].lower().endswith("ed") and len(tokens[i]) > 3

    tagger.add_rule(TransformationRule(
        rule_id="RULE_03",
        description="Change NN -> VBD if word ends in 'ed'",
        from_tag="NN",
        to_tag="VBD",
        condition=cond_ends_ed
    ))

    # Rule 4: Change NN to VBG if word ends in 'ing' (e.g. "running", "fishing")
    def cond_ends_ing(tokens, tags, i):
        return tokens[i].lower().endswith("ing") and len(tokens[i]) > 4

    tagger.add_rule(TransformationRule(
        rule_id="RULE_04",
        description="Change NN -> VBG if word ends in 'ing'",
        from_tag="NN",
        to_tag="VBG",
        condition=cond_ends_ing
    ))

    # Rule 5: Change NN to RB if word ends in 'ly' (e.g. "quickly", "loudly")
    def cond_ends_ly(tokens, tags, i):
        return tokens[i].lower().endswith("ly") and len(tokens[i]) > 3

    tagger.add_rule(TransformationRule(
        rule_id="RULE_05",
        description="Change NN -> RB if word ends in 'ly'",
        from_tag="NN",
        to_tag="RB",
        condition=cond_ends_ly
    ))

    # Rule 6: Change NN to JJ if next tag is NN/NNS (noun pre-modifier) (e.g. "apple cake")
    def cond_noun_before_noun(tokens, tags, i):
        return i + 1 < len(tags) and tags[i + 1] in ("NN", "NNS")

    tagger.add_rule(TransformationRule(
        rule_id="RULE_06",
        description="Change NN -> JJ if followed by another Noun (NN)",
        from_tag="NN",
        to_tag="JJ",
        condition=cond_noun_before_noun
    ))


def main():
    print("=" * 80)
    print("EXPERIMENT 10: TRANSFORMATION-BASED TAGGING (TBL / BRILL ALGORITHM)")
    print("=" * 80)

    tagger = TransformationBasedTagger(default_tag="NN")
    setup_sample_rules(tagger)

    print("Configured Transformation Rules:")
    for r in tagger.rules:
        print(f"  [{r.rule_id}] {r.description}")

    sample_sentences = [
        ["They", "will", "fish", "in", "the", "lake", "."],
        ["He", "kicked", "the", "ball", "quickly", "."],
        ["She", "can", "program", "an", "apple", "pie", "recipe", "."],
        ["The", "running", "water", "was", "calm", "."]
    ]

    for sent_idx, tokens in enumerate(sample_sentences, 1):
        print("\n" + "=" * 80)
        print(f"SENTENCE #{sent_idx}: {' '.join(tokens)}")
        print("=" * 80)

        final_tags, history = tagger.tag_with_trace(tokens)

        # Baseline display
        base_step = history[0]
        print(f"\nStep 0: Baseline Initialization:")
        for w, t in zip(tokens, base_step[1]):
            print(f"   {w:<10} -> {t}")

        # Applied rules steps
        if len(history) > 1:
            print("\nRule Transformations Applied:")
            for step_id, current_tags_state, changes in history[1:]:
                rule_obj = next(r for r in tagger.rules if r.rule_id == step_id)
                print(f"\n * Triggered {step_id}: {rule_obj.description}")
                for idx, word, old_t, new_t in changes:
                    print(f"   -> Word '{word}' at index {idx}: Changed {old_t} ==> {new_t}")

        print("\nFinal Tagged Sentence:")
        print(" ", " ".join([f"{w}/{t}" for w, t in zip(tokens, final_tags)]))


if __name__ == "__main__":
    main()
