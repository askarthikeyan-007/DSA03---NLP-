"""
================================================================================
EXPERIMENT 24
Title: Dialog Act Recognition in Conversational Systems
Objective: Implement a Python program that recognizes Dialog Acts (Speech Acts)
           such as Greeting, Question, Statement, Request, Agreement, and Closing
           in a multi-turn conversation using linguistic and structural cues.
================================================================================
"""

import re
from typing import Tuple


class DialogActRecognizer:
    """
    Classifies conversational utterances into Speech / Dialog Acts:
      - GREETING
      - QUESTION (Wh-question or Yes/No question)
      - REQUEST / DIRECTIVE
      - STATEMENT / INFORM
      - OPINION
      - AGREEMENT / ACCEPT
      - DISAGREEMENT / REJECT
      - APOLOGY / THANKING
      - CLOSING / FAREWELL
    """

    def __init__(self):
        # Trigger patterns for each dialog act category
        self.rules = [
            ("GREETING", r"^(hi|hello|hey|good\s+(morning|afternoon|evening)|greetings)\b"),
            ("CLOSING", r"^(bye|goodbye|see\s+you|farewell|have\s+a\s+(good|nice)\s+day|take\s+care)\b"),
            ("THANKING", r"\b(thank\s+you|thanks|much\s+appreciated|grateful)\b"),
            ("APOLOGY", r"\b(sorry|apologies|pardon\s+me|excuse\s+me)\b"),
            ("AGREEMENT", r"^(yes|yeah|yep|sure|absolutely|correct|indeed|agreed|sounds\s+good|ok|okay)\b"),
            ("DISAGREEMENT", r"^(no|nope|not\s+really|i\s+disagree|incorrect|wrong|never)\b"),
            ("REQUEST", r"^(please|could\s+you|would\s+you|can\s+you|kindly|let\s+me|show\s+me)\b"),
            ("QUESTION", r"^(what|when|where|which|who|whom|whose|why|how|is|are|was|were|do|does|did|can|could|will|would)\b|\?$"),
            ("OPINION", r"\b(i\s+(think|believe|feel|guess)|in\s+my\s+opinion|seems\s+to\s+me|probably)\b"),
        ]

    def classify_utterance(self, utterance: str) -> Tuple[str, str]:
        """Classifies an individual utterance and returns (dialog_act, cue_matched)."""
        clean = utterance.strip()
        lower = clean.lower()

        # Check explicit rules
        for act, pattern in self.rules:
            match = re.search(pattern, lower)
            if match:
                return act, match.group()

        # Fallback heuristic: ends with '?' is Question
        if clean.endswith("?"):
            return "QUESTION", "?"

        # Default fallback is an informative statement
        return "STATEMENT", "declarative sentence structure"


def main():
    print("=" * 80)
    print("EXPERIMENT 24: DIALOG ACT RECOGNITION IN CONVERSATIONS")
    print("=" * 80)

    # Multi-turn conversation between Customer and Virtual Support Assistant
    conversation = [
        ("Customer", "Hello! Good morning."),
        ("Agent",    "Hi there! How can I assist you today?"),
        ("Customer", "Could you check the status of my order #54321?"),
        ("Agent",    "Sure, let me check the database for you."),
        ("Agent",    "Your order has been shipped and will arrive tomorrow."),
        ("Customer", "I think that delivery is faster than expected!"),
        ("Agent",    "Yes, our priority shipping team dispatched it earlier."),
        ("Customer", "I am very sorry for troubling you early in the morning."),
        ("Agent",    "No problem at all! Is there anything else you need?"),
        ("Customer", "No, that is all. Thank you very much!"),
        ("Agent",    "You are welcome! Have a wonderful day. Goodbye!"),
        ("Customer", "Bye!")
    ]

    recognizer = DialogActRecognizer()

    print(f"{'Speaker':<10} | {'Utterance':<50} | {'Dialog Act':<15} | {'Trigger Cue'}")
    print("-" * 95)

    for speaker, utterance in conversation:
        act, cue = recognizer.classify_utterance(utterance)
        utt_display = utterance if len(utterance) <= 48 else utterance[:45] + "..."
        print(f"{speaker:<10} | {utt_display:<50} | {act:<15} | '{cue}'")

    print("-" * 95)


if __name__ == "__main__":
    main()
