"""
================================================================================
EXPERIMENT 01
Title: Regular Expressions for Pattern Matching and Searching in Text
Objective: Demonstrate how to use regular expressions in Python using the `re`
           module to search, match, extract, substitute, and split patterns.
================================================================================
"""

import re


def demonstrate_regex():
    sample_text = (
        "Hello! Contact us at support@example.com or sales.dept@nlp-corp.org. "
        "Call +1-800-555-0199 or +91-9876543210 on 2026-11-16. "
        "Product prices are $199.99 and $49.50. "
        "Visit https://www.nlpdomain.com for more info."
    )

    print("=" * 70)
    print("EXPERIMENT 01: REGULAR EXPRESSIONS IN PYTHON")
    print("=" * 70)
    print("Sample Text:\n", sample_text)
    print("-" * 70)

    # 1. re.search: Search for the first occurrence of a pattern
    print("\n[1] re.search (Locate first phone number):")
    phone_pattern = r"\+?\d{1,3}[-\s]?\d{3,4}[-\s]?\d{4,6}"
    search_result = re.search(phone_pattern, sample_text)
    if search_result:
        print(f"    Found: '{search_result.group()}' at span {search_result.span()}")

    # 2. re.match: Check for match at the beginning of the string
    print("\n[2] re.match (Check start of text):")
    match_result = re.match(r"^Hello!", sample_text)
    if match_result:
        print(f"    Text starts with: '{match_result.group()}'")
    else:
        print("    No match at the beginning of the string.")

    # 3. re.findall: Extract all occurrences
    print("\n[3] re.findall (Extract structured entities):")

    # Email extraction
    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+"
    emails = re.findall(email_pattern, sample_text)
    print(f"    Emails found ({len(emails)}): {emails}")

    # Phone numbers extraction
    all_phones = re.findall(phone_pattern, sample_text)
    print(f"    Phone numbers found ({len(all_phones)}): {all_phones}")

    # Prices extraction
    price_pattern = r"\$\d+\.\d{2}"
    prices = re.findall(price_pattern, sample_text)
    print(f"    Prices found ({len(prices)}): {prices}")

    # Date extraction (YYYY-MM-DD)
    date_pattern = r"\b\d{4}-\d{2}-\d{2}\b"
    dates = re.findall(date_pattern, sample_text)
    print(f"    Dates found ({len(dates)}): {dates}")

    # URL extraction
    url_pattern = r"https?://[^\s]+"
    urls = re.findall(url_pattern, sample_text)
    print(f"    URLs found ({len(urls)}): {urls}")

    # 4. re.finditer: Iterate over match objects with position info
    print("\n[4] re.finditer (Detailed match positions for prices):")
    for idx, match in enumerate(re.finditer(price_pattern, sample_text), 1):
        print(f"    Match #{idx}: {match.group()} -> Start: {match.start()}, End: {match.end()}")

    # 5. re.sub: Pattern replacement (Text Normalization / Redaction)
    print("\n[5] re.sub (Mask sensitive emails and phone numbers):")
    masked_text = re.sub(email_pattern, "[REDACTED_EMAIL]", sample_text)
    masked_text = re.sub(phone_pattern, "[REDACTED_PHONE]", masked_text)
    print("    Masked Text:\n   ", masked_text)

    # 6. re.split: Tokenization and splitting
    print("\n[6] re.split (Sentence / Word Tokenization):")
    words = re.split(r"[^\w]+", sample_text)
    filtered_words = [w for w in words if w]
    print(f"    First 10 tokenized words: {filtered_words[:10]}")


if __name__ == "__main__":
    demonstrate_regex()
