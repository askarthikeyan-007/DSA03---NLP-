"""
================================================================================
EXPERIMENT 25
Title: Text Generation using GPT-3 (OpenAI API)
Objective: Implement a Python program using the official OpenAI library to
           generate coherent text completions from a user prompt, demonstrating
           API integration, prompt engineering, and hyperparameter tuning.
================================================================================
"""

import os
import sys

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def generate_with_openai(prompt: str, api_key: str = None, temperature: float = 0.7, max_tokens: int = 150):
    """
    Invokes OpenAI's generative language model API.
    If no valid API key is configured, provides an offline simulation
    demonstrating the exact prompt-response pipeline.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY")

    print("\n--- OpenAI Model Generation Request ---")
    print(f"Model Selected   : gpt-3.5-turbo / gpt-3 completion")
    print(f"Temperature      : {temperature} (Controls randomness: 0=deterministic, 1=creative)")
    print(f"Max Tokens       : {max_tokens} (Upper limit on output length)")
    print(f"User Prompt      : \"{prompt}\"")
    print("-" * 80)

    if key and key != "YOUR_OPENAI_API_KEY":
        try:
            client = OpenAI(api_key=key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a knowledgeable and concise NLP assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            generated_text = response.choices[0].message.content
            print("API Response (Live Model Output):\n")
            print(generated_text)
            return generated_text
        except Exception as e:
            print(f"[!] OpenAI API request failed: {e}")
            print("Falling back to simulated completion mode...\n")

    # Simulation fallback for offline lab evaluation without a paid API key
    print("[NOTE] No active OpenAI API key detected in environment variable OPENAI_API_KEY.")
    print("To run with live OpenAI servers:")
    print("  1. Obtain an API key from https://platform.openai.com/")
    print("  2. Set environment variable: set OPENAI_API_KEY=\"sk-...\"")
    print("\nDemonstrating simulated output for educational lab evaluation:")

    simulated_responses = {
        "Explain the importance of Natural Language Processing in 3 sentences.": (
            "Natural Language Processing bridges the communication gap between human language "
            "and computational systems, enabling machines to understand, interpret, and generate text. "
            "It powers critical modern applications including search engines, voice assistants, and automated translation. "
            "Ultimately, NLP empowers computers to extract valuable insights from vast volumes of unstructured textual data."
        ),
        "Write a haiku about artificial intelligence.": (
            "Lines of silent code,\n"
            "Thinking minds in silicon,\n"
            "Future awakens."
        )
    }

    completion = simulated_responses.get(
        prompt,
        f"Simulated completion for prompt '{prompt}': Large language models synthesize patterns "
        f"learned from extensive pretraining corpora to generate fluent, context-appropriate responses."
    )
    print("Generated Completion:\n")
    print(completion)
    return completion


def main():
    print("=" * 80)
    print("EXPERIMENT 25: TEXT GENERATION USING OPENAI GPT-3 / GPT-3.5")
    print("=" * 80)

    prompts = [
        "Explain the importance of Natural Language Processing in 3 sentences.",
        "Write a haiku about artificial intelligence."
    ]

    for p in prompts:
        generate_with_openai(prompt=p, temperature=0.7, max_tokens=100)
        print("=" * 80)


if __name__ == "__main__":
    main()
