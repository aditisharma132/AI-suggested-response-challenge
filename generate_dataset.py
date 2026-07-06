"""
Generates a synthetic customer support email dataset using the free Gemini API.

Setup:
    pip install google-generativeai python-dotenv
    Create a .env file with: GEMINI_API_KEY=your_key_here

Run:
    python generate_dataset.py

Output:
    dataset.json — list of {customer_email, agent_reply} pairs, the "past emails and
    responses" the generator will retrieve from / few-shot on.
"""

import os
import json
import time
import re
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY in a .env file first.")

genai.configure(api_key=API_KEY)

valid_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
model_name = "gemini-1.5-flash"
if f"models/{model_name}" not in valid_models:
    # Fallback to the first available gemini flash or pro model
    fallback = next((m for m in valid_models if "flash" in m), None) or next((m for m in valid_models if "pro" in m), "gemini-pro")
    model_name = fallback.replace("models/", "")

print(f"Auto-selected model: {model_name}")
model = genai.GenerativeModel(model_name)

CATEGORIES = [
    "refund_request",
    "shipping_delay",
    "billing_dispute",
    "angry_escalation",
    "simple_faq",
    "account_access",
    "product_defect",
]

DIFFICULTIES = ["easy", "medium", "hard"]

# 7 categories x 3 difficulties x 3 each ≈ 63 pairs — enough diversity, keeps generation fast
BATCH_SIZE = 3

PROMPT_TEMPLATE = """Generate {n} realistic customer support email THREADS for an e-commerce/SaaS company.
Each thread = one customer email PLUS the agent's actual reply that was sent (as if pulled from a real support inbox history).

Category: {category}
Difficulty: {difficulty}
  - easy: single clear intent, calm tone, short
  - medium: some ambiguity or a secondary request, moderate length
  - hard: messy/rambling, mixed intents, typos, high emotion, or missing key info

Make each customer email sound like a real person wrote it — vary sentence structure, length, and tone.
Include realistic invented details (order numbers, dates, product names).

The agent_reply should be a GOOD, realistic reply a competent support agent would actually send:
specific to the customer's email (references their actual details), empathetic where warranted,
resolves or clearly progresses the issue, professional but warm, and reasonably concise (not a wall of text).

Return ONLY a JSON array (no markdown fences, no commentary) where each item has exactly these fields:
{{
  "category": "{category}",
  "difficulty": "{difficulty}",
  "customer_email_text": "...",
  "customer_sentiment": "one of: neutral, frustrated, angry, confused, satisfied, urgent",
  "context_notes": "short note on prior interactions or urgency, can be empty string",
  "agent_reply": "the good reply that was actually sent, grounded in the specific email"
}}
"""


def clean_json_array(text):
    """Strip markdown fences and extract the JSON array if the model wraps it."""
    text = text.strip()
    text = re.sub(r"^```json\s*|^```\s*|```$", "", text, flags=re.MULTILINE).strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        text = match.group(0)
    return text


def generate_batch(category, difficulty, n=BATCH_SIZE, retries=3):
    prompt = PROMPT_TEMPLATE.format(n=n, category=category, difficulty=difficulty)
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            raw = clean_json_array(response.text)
            items = json.loads(raw)
            return items
        except Exception as e:
            print(f"  retry {attempt+1}/{retries} for {category}/{difficulty}: {e}")
            time.sleep(2)
    print(f"  FAILED: {category}/{difficulty} after {retries} retries")
    return []


def main():
    dataset = []
    next_id = 1

    for category in CATEGORIES:
        for difficulty in DIFFICULTIES:
            print(f"Generating {category} / {difficulty} ...")
            items = generate_batch(category, difficulty)
            for item in items:
                item["id"] = f"email_{next_id:03d}"
                next_id += 1
                dataset.append(item)
            time.sleep(4.5)  # strictly enforces the 15 RPM free tier limit

    with open("dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\nDone. Wrote {len(dataset)} examples to dataset.json")


if __name__ == "__main__":
    main()
