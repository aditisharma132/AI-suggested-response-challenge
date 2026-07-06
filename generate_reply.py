"""
Given a new incoming customer email, retrieves the most similar past
email/reply pairs from dataset.json and uses them as few-shot grounding
to generate a suggested reply via Gemini.

Why retrieval + few-shot instead of fine-tuning:
- Fine-tuning needs far more data than 60 pairs to actually generalize, and
  costs setup time we don't have in a 100-minute window.
- Retrieval + few-shot grounds the model in *this company's* actual tone and
  resolution patterns without training a model, and is easy to inspect/debug
  (you can see exactly which past examples influenced a given reply).
- Trade-off: quality depends on retrieval quality and dataset coverage. If no
  similar past case exists, the model falls back to general reasoning, which
  is flagged in the output.

Setup:
    pip install google-generativeai python-dotenv scikit-learn
    Requires dataset.json (run generate_dataset.py first)

Run:
    python generate_reply.py --email "path/to/incoming_email.txt"
    or
    python generate_reply.py --text "Hi, my order #123 hasn't arrived..."
"""

import os
import json
import argparse
import google.generativeai as genai
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

SYSTEM_PROMPT = """You are a customer support agent for an e-commerce/SaaS company.
Write replies that are:
- Empathetic where the customer shows frustration or urgency
- Accurate: only reference details actually present in the customer's email, never invent order details, refund amounts, or policies not given to you
- Actionable: give a clear next step or resolution, not vague reassurance
- Concise: no filler, no unnecessary repetition of the customer's message
- Professional but warm in tone

If the conversation has gone back and forth multiple times and the customer remains highly frustrated, OR if they explicitly demand a phone call or manager, you MUST gracefully escalate by stating exactly: "I will now escalate this to a human agent. Please expect a follow-up shortly." and nothing else.

You will be shown a few past examples of similar customer emails and the replies that were sent.
Use them to match the company's tone and resolution style, but write a reply specific to the NEW Email or Thread below — do not copy details from the examples."""


def load_dataset(path="dataset.json"):
    with open(path) as f:
        return json.load(f)


def retrieve_similar(new_email_text, dataset, k=3):
    """TF-IDF cosine similarity retrieval — simple, fast, no extra API calls."""
    corpus = [item["customer_email_text"] for item in dataset]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(corpus + [new_email_text])
    sims = cosine_similarity(tfidf[-1], tfidf[:-1]).flatten()
    top_k_idx = sims.argsort()[::-1][:k]
    return [(dataset[i], sims[i]) for i in top_k_idx]


def build_prompt(new_email_text, retrieved):
    examples_block = ""
    for i, (item, score) in enumerate(retrieved, 1):
        examples_block += f"""
Example {i} (similarity: {score:.2f}):
Customer email: {item['customer_email_text']}
Agent reply: {item['agent_reply']}
"""

    return f"""{SYSTEM_PROMPT}

--- PAST EXAMPLES ---
{examples_block}
--- NEW CUSTOMER EMAIL/THREAD TO REPLY TO ---
{new_email_text}

Write only the reply text, no preamble, no labels."""


def generate_reply(new_email_text, dataset_path="dataset.json", k=3):
    dataset = load_dataset(dataset_path)
    retrieved = retrieve_similar(new_email_text, dataset, k=k)
    prompt = build_prompt(new_email_text, retrieved)
    response = model.generate_content(prompt)
    return {
        "incoming_email": new_email_text,
        "retrieved_examples": [
            {"category": item["category"], "similarity": round(float(score), 3)}
            for item, score in retrieved
        ],
        "generated_reply": response.text.strip(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", help="Path to a text file with the incoming email")
    parser.add_argument("--text", help="Incoming email text directly")
    args = parser.parse_args()

    if args.email:
        with open(args.email) as f:
            text = f.read()
        result = generate_reply(text)
        print(json.dumps(result, indent=2))
    elif args.text:
        result = generate_reply(args.text)
        print(json.dumps(result, indent=2))
    else:
        print("--- Starting Interactive Support Thread ---")
        print("Type your message as the customer. The AI will reply.")
        print("The thread continues until the AI decides to escalate to a human/call.")
        print("Type 'exit' to manually stop.\n")
        
        conversation_history = ""
        while True:
            user_input = input("Customer: ")
            if user_input.lower() in ["quit", "exit"]:
                break
                
            conversation_history += f"Customer: {user_input}\n"
            
            result = generate_reply(conversation_history)
            reply = result["generated_reply"]
            
            print(f"\nAI Agent: {reply}\n")
            conversation_history += f"Agent: {reply}\n"
            
            # Break the loop if the bot escalates
            escaped_lower = reply.lower()
            if "escalate" in escaped_lower and "human" in escaped_lower:
                print("--- Thread gracefully escalated and stopped. ---")
                break
