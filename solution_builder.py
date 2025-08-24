# solution_builder.py — Google Gemini 2.5 Flash

import os
import argparse
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm
import time, random

# --------------------------
# Config
# --------------------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not found in .env or environment")
genai.configure(api_key=API_KEY)


# --------------------------
# LLM Client
# --------------------------
class SolutionBuilder:
    def __init__(self, model="gemini-2.5-flash"):
        self.model = genai.GenerativeModel(model)

    def generate(self, question: str, similar_context: str = None) -> str:
        """Generate a solution. If similar_context is provided, include it in the prompt."""
        if similar_context:
            prompt = f"""
            Solve the following QUESTION using help from SIMILAR QUESTIONS and their approaches.

            QUESTION:
            {question}

            SIMILAR QUESTIONS & APPROACHES:
            {similar_context}

            Provide a clear and concise step-by-step solution.
            """
        else:
            prompt = f"""
            Solve the following QUESTION independently (without external context).

            QUESTION:
            {question}

            Provide a clear and concise step-by-step solution.
            """

        for attempt in range(3):
            try:
                response = self.model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                wait = (2 ** attempt) + random.random()
                print(f"⚠️ Error: {e}. Retrying in {wait:.1f}s...")
                time.sleep(wait)

        return "Error: solution generation failed after retries."


# --------------------------
# Main Function
# --------------------------
def build_solutions(input_csv: str, out_csv: str, limit: int = None):
    df = pd.read_csv(input_csv)
    if limit:
        df = df.head(limit)

    builder = SolutionBuilder()
    results = []

    # Group by unique question ID so we can gather ALL similars
    grouped = df.groupby("id")

    for qid, group in tqdm(grouped, total=len(grouped), desc="Building solutions"):
        q = group["question"].iloc[0]
        subject = group["subject"].iloc[0]

        # Build combined context
        similar_context = ""
        for _, row in group.iterrows():
            sim_q = row.get("similar_question", "")
            sim_app = row.get("approach_summary", "")
            if pd.notna(sim_q) and pd.notna(sim_app):
                similar_context += f"\n- Similar Q: {sim_q}\n  Approach: {sim_app}"

        # Generate baseline + augmented solutions
        baseline_sol = builder.generate(q)
        augmented_sol = builder.generate(q, similar_context if similar_context else None)

        results.append({
            "id": qid,
            "subject": subject,
            "question": q,
            "baseline_solution": baseline_sol,
            "augmented_solution": augmented_sol,
            "similar_context": similar_context.strip()
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"✅ Solutions saved to {out_csv}")


# --------------------------
# CLI Entrypoint
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build solutions with and without similar questions")
    parser.add_argument("--input", type=str, default="runs/google_results_all.csv", help="Input CSV from relevance.py")
    parser.add_argument("--output", type=str, default="runs/solutions.csv", help="Output CSV with solutions")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions (to save credits)")
    args = parser.parse_args()

    build_solutions(args.input, args.output, args.limit)
