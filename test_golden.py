import time
import json
import webbrowser
import os
from generate_reply import generate_reply
from evaluate import score_reply

# The 3 hard-coded emails from our golden_sets.md document
GOLDEN_EMAILS = [
    {
        "name": "Angry Escalation",
        "email": "THIS IS UNACCEPTABLE. I've emailed 3 times and nobody has fixed my sync issue. I am losing money every hour your app is down. I want a manager to call me NOW."
    },
    {
        "name": "Complex Billing Dispute",
        "email": "I see two charges of $49.99 on my statement for this month. I only have one account. Did you double charge me?"
    },
    {
        "name": "High-Urgency Delivery Failure",
        "email": "Where is my order??? The tracking hasn't updated in 4 days. It was supposed to be here for my daughter's birthday tomorrow."
    }
]

def test_golden_set():
    print("=== RUNNING GOLDEN SET END-TO-END EVALUATION ===")
    print("Testing the pipeline: RAG Retrieval -> AI Generation -> LLM-as-a-judge Scoring\n")
    
    total_score = 0
    results = []
    
    for i, test_case in enumerate(GOLDEN_EMAILS[:1], 1):
        print(f"\n--- Test Case {i}: {test_case['name']} ---")
        print(f"Customer Email: {test_case['email']}")
        
        # 1. Generate the Reply
        print("\n> Generating reply via RAG...")
        start_time = time.time()
        reply_data = generate_reply(test_case['email'], k=3)
        generated_text = reply_data["generated_reply"]
        print(f"Agent Reply ({time.time() - start_time:.1f}s):\n{generated_text}\n")
        
        time.sleep(5) # Strict 5s polite delay to bypass 15 RPM limits
        
        # 2. Evaluate the Reply
        print("> Evaluating reply quality using LLM judge...")
        scores = score_reply(test_case['email'], generated_text)
        
        if scores:
            overall = scores.get("overall", 0)
            total_score += overall
            print(f"Scores -> Correctness: {scores['correctness']['score']}/5 | Empathy: {scores['empathy']['score']}/5 | Actionability: {scores['actionability']['score']}/5")
            print(f"Judge Reason (Correctness): {scores['correctness']['reason']}")
            print(f"OVERALL SCORE: {overall}/5.0")
            print("-" * 50)
            
            results.append(overall)
        else:
            print("Evaluation failed (possible rate limit).")
            break
            
        time.sleep(2) # Politeness to rate limits before next generation call
        
    if results:
        avg_score = total_score / len(results)
        print(f"\n======================================")
        print(f"GOLDEN SET AVERAGE SCORE: {avg_score:.2f} / 5.0")
        print(f"======================================")
        
        # Generate and open a visual HTML dashboard!
        print("\nGenerating visual dashboard on a new browser tab...")
        color = "#22c55e" if avg_score >= 4.0 else "#ef4444"
        status = "READY TO PUBLISH" if avg_score >= 4.0 else "REQUIRES TUNING"
        
        html_content = f"""
        <html>
        <head>
            <title>AI Evaluation Complete</title>
            <style>
                body {{ font-family: -apple-system, sans-serif; background: #0f172a; color: white; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }}
                .card {{ background: #1e293b; padding: 50px; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.8); text-align: center; border-top: 8px solid {color}; }}
                h1 {{ font-size: 5rem; margin: 0 0 10px 0; color: {color}; }}
                h2 {{ margin: 0; color: #94a3b8; font-weight: normal; font-size: 1.5rem; }}
                .badge {{ display: inline-block; padding: 10px 20px; background: rgba(255,255,255,0.1); border-radius: 30px; margin-top: 20px; font-weight: bold; letter-spacing: 2px; font-size: 1.2rem; }}
            </style>
        </head>
        <body>
            <div class="card">
                <h2>AI Golden Set Evaluation Metric</h2>
                <h1>{avg_score:.2f} / 5.0</h1>
                <div class="badge" style="color: {color};">{status}</div>
                <p style="color: #64748b; font-size: 1rem; margin-top: 30px;">Successfully scored {len(results)} exact-match test cases</p>
            </div>
        </body>
        </html>
        """
        
        html_path = os.path.abspath("accuracy_report.html")
        with open(html_path, "w") as f:
            f.write(html_content)
            
        webbrowser.open_new_tab(f"file://{html_path}")
            
if __name__ == "__main__":
    test_golden_set()
