"""
Accuracy/evaluation system for generated support email replies.

Why not exact match / BLEU / ROUGE:
Exact match is meaningless here — there are many equally good ways to phrase
a correct, helpful reply, and the "reference" reply in our dataset is just
ONE good reply, not the only acceptable one. String-overlap metrics (BLEU,
ROUGE) reward copying the reference's wording, not correctness or usefulness,
and would penalize a differently-worded but equally good reply. So we score
against a rubric of what actually makes a reply good, using an LLM judge,
similar to how a real QA reviewer would grade a support ticket.

Rubric dimensions (1-5 each):
  - relevance:     does it address what the customer actually asked?
  - correctness:   does it avoid inventing facts/policies not in the email?
  - empathy:       does tone match the customer's emotional state?
  - actionability: does it give a concrete next step or resolution?
  - conciseness:   is it appropriately brief, no filler/repetition?

Weights (overall = weighted average, 0-5 scale):
  correctness   0.30  -- a factually wrong reply is actively harmful, weighted highest
  relevance     0.25  -- must address the actual ask
  actionability 0.20  -- a correct but vague reply still fails the customer
  empathy       0.15  -- matters but is secondary to getting the substance right
  conciseness   0.10  -- nice-to-have, least likely to cause real harm if imperfect

Validation approach:
This script includes a `--validate` mode that runs the judge on a small sample
of the dataset's OWN reference replies (which we assume are already "good").
If the judge scores those references consistently high (e.g. avg >= 4.0), that's
evidence the rubric isn't miscalibrated or overly harsh/lenient in a way that
would make its scores meaningless. This is a spot-check, not a full validation --
documented honestly in the README as a limitation.

Setup:
    pip install google-generativeai python-dotenv

Run:
    python evaluate.py --email "..." --reply "..."
    python evaluate.py --batch dataset.json --generated generated_replies.json
    python evaluate.py --validate dataset.json
"""

import os
import json
import argparse
import re
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY in a .env file first.")

genai.configure(api_key=API_KEY)

valid_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
model_name = "gemini-1.5-flash"
if f"models/{model_name}" not in valid_models:
    fallback = next((m for m in valid_models if "flash" in m), None) or next((m for m in valid_models if "pro" in m), "gemini-pro")
    model_name = fallback.replace("models/", "")

print(f"Auto-selected model: {model_name}")
model = genai.GenerativeModel(model_name)

WEIGHTS = {
    "correctness": 0.30,
    "relevance": 0.25,
    "actionability": 0.20,
    "empathy": 0.15,
    "conciseness": 0.10,
}

JUDGE_PROMPT = """You are a strict, experienced customer support QA reviewer.
Score the AGENT REPLY below against the CUSTOMER EMAIL on these 5 dimensions,
each on a 1-5 integer scale:

- relevance (1=ignores the actual request, 5=fully addresses everything asked)
- correctness (1=invents facts/policies/details not given, 5=makes no unsupported claims)
- empathy (1=tone mismatched to customer's emotional state, 5=tone well-matched)
- actionability (1=vague, no clear next step, 5=clear concrete resolution/next step)
- conciseness (1=rambling/repetitive, 5=appropriately brief with no filler)

For each dimension, also give a one-sentence reason.

CUSTOMER EMAIL:
{email}

AGENT REPLY (to be scored):
{reply}

Return ONLY valid JSON, no markdown fences, no commentary, in exactly this format:
{{
  "relevance": {{"score": 0, "reason": "..."}},
  "correctness": {{"score": 0, "reason": "..."}},
  "empathy": {{"score": 0, "reason": "..."}},
  "actionability": {{"score": 0, "reason": "..."}},
  "conciseness": {{"score": 0, "reason": "..."}}
}}
"""


def clean_json(text):
    text = re.sub(r"^```json\s*|^```\s*|```$", "", text.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text


def score_reply(customer_email, agent_reply, retries=3):
    prompt = JUDGE_PROMPT.format(email=customer_email, reply=agent_reply)
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            scores = json.loads(clean_json(response.text))
            overall = sum(scores[dim]["score"] * w for dim, w in WEIGHTS.items())
            scores["overall"] = round(overall, 2)
            return scores
        except Exception as e:
            print(f"  retry {attempt+1}/{retries}: {e}")
    return None


def run_single(email, reply):
    result = score_reply(email, reply)
    print(json.dumps(result, indent=2))
    return result


def run_batch(dataset_path, generated_path):
    with open(dataset_path) as f:
        dataset = json.load(f)
    with open(generated_path) as f:
        generated = json.load(f)  # list of {"id" or "customer_email_text", "generated_reply"}

    results = []
    for item in generated:
        email_text = item.get("incoming_email") or item.get("customer_email_text")
        reply_text = item.get("generated_reply")
        scores = score_reply(email_text, reply_text)
        results.append({
            "email": email_text[:80] + "...",
            "reply": reply_text[:80] + "...",
            "scores": scores,
        })

    overall_scores = [r["scores"]["overall"] for r in results if r["scores"]]
    summary = {
        "per_response": results,
        "overall_system_score": round(sum(overall_scores) / len(overall_scores), 2) if overall_scores else None,
        "n_scored": len(overall_scores),
    }
    with open("evaluation_report.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Scored {len(overall_scores)} replies. Overall system score: {summary['overall_system_score']}")
    print("Full report written to evaluation_report.json")


def run_validation(dataset_path, sample_size=10):
    """Sanity-check the judge against the dataset's own reference replies,
    which we treat as a 'known good' baseline."""
    with open(dataset_path) as f:
        dataset = json.load(f)
    sample = dataset[:sample_size]

    overalls = []
    print(f"Validating judge against {len(sample)} known-good reference replies...\n")
    for item in sample:
        scores = score_reply(item["customer_email_text"], item["agent_reply"])
        if scores:
            overalls.append(scores["overall"])
            print(f"  [{item['category']}/{item['difficulty']}] overall={scores['overall']}")

    avg = sum(overalls) / len(overalls) if overalls else 0
    print(f"\nAverage overall score on reference replies: {avg:.2f} / 5.0")
    if avg >= 4.0:
        print("=> Judge scores known-good replies highly. Rubric appears reasonably calibrated.")
    else:
        print("=> Judge is scoring reference replies lower than expected — rubric or prompt "
              "may be too strict, or reference replies have real flaws worth reviewing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", help="Customer email text (single mode)")
    parser.add_argument("--reply", help="Generated reply text (single mode)")
    parser.add_argument("--batch", help="Path to dataset.json (batch mode)")
    parser.add_argument("--generated", help="Path to a JSON file of generated replies (batch mode)")
    parser.add_argument("--validate", help="Path to dataset.json to run validation mode")
    args = parser.parse_args()

    if args.validate:
        run_validation(args.validate)
    elif args.batch and args.generated:
        run_batch(args.batch, args.generated)
    elif args.email and args.reply:
        run_single(args.email, args.reply)
    else:
        print("Usage:\n"
              "  python evaluate.py --email '...' --reply '...'\n"
              "  python evaluate.py --batch dataset.json --generated generated_replies.json\n"
              "  python evaluate.py --validate dataset.json")
