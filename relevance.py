# relevance.py — Google Gemini (2.5-flash) only, batched, CSV in/out (paid-friendly)

import os
import json
import time
import random
import argparse
from typing import List, Dict, Any

import pandas as pd
import google.generativeai as genai
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from tqdm import tqdm

# --------------------------
# Config & API key
# --------------------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not set in .env or environment")
genai.configure(api_key=API_KEY)


# --------------------------
# Schema
# --------------------------
class RelevanceScore(BaseModel):
    conceptual: int = Field(..., ge=0, le=4)
    structural: int = Field(..., ge=0, le=4)
    difficulty: int = Field(..., ge=0, le=4)
    transferability: int = Field(..., ge=0, le=4)
    overall: float = Field(..., ge=0, le=4)
    rationale: str


# --------------------------
# JSON parsing helpers
# --------------------------
def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        if t.lower().startswith("json"):
            t = t[4:].lstrip()
    return t


def _parse_json_list_strict(text: str) -> List[Dict[str, Any]]:
    """Parse model output into a JSON array, handling code fences/wrappers."""
    t = _strip_code_fences(text)
    try:
        data = json.loads(t)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # fallback: extract [ ... ]
    start, end = t.find("["), t.rfind("]")
    if start != -1 and end != -1 and end > start:
        snippet = t[start:end+1]
        data = json.loads(snippet)
        if isinstance(data, list):
            return data

    raise ValueError("Model did not return a valid JSON array.")


# --------------------------
# LLM client (Google only)
# --------------------------
class LLMClient:
    def __init__(self, model: str = "gemini-2.5-flash", max_retries: int = 5, base_delay: float = 2.0):
        self.model_name = model
        self.client = genai.GenerativeModel(model)
        self.max_retries = max_retries
        self.base_delay = base_delay

    def evaluate_batch(self, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate multiple question pairs at once.
        """
        header = (
            "You are a strict evaluator of question similarity.\n"
            "For EACH pair below, score similarity on:\n"
            "- conceptual (0-4)\n- structural (0-4)\n- difficulty (0-4)\n- transferability (0-4)\n"
            "- overall (float, 0-4)\nInclude a concise rationale.\n\n"
            "Return ONLY a JSON array with the SAME number of items as pairs, in the SAME order.\n"
            "Schema per item: {\"conceptual\":int,\"structural\":int,\"difficulty\":int,"
            "\"transferability\":int,\"overall\":float,\"rationale\":str}\n\n"
        )
        body = []
        for i, p in enumerate(pairs, start=1):
            body.append(
                f"PAIR {i}:\nQUESTION: {p['question']}\nSIMILAR QUESTION: {p['similar_question']}\n"
            )
        prompt = header + "\n".join(body)

        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.generate_content(prompt)
                items = _parse_json_list_strict(resp.text)

                # pad or truncate if mismatch
                if len(items) < len(pairs):
                    items.extend([
                        {"conceptual": 0, "structural": 0, "difficulty": 0,
                         "transferability": 0, "overall": 0.0,
                         "rationale": "Model returned fewer items than expected."}
                        for _ in range(len(pairs) - len(items))
                    ])
                elif len(items) > len(pairs):
                    items = items[:len(pairs)]

                results = []
                for p, s in zip(pairs, items):
                    try:
                        score = RelevanceScore(**s)
                    except Exception:
                        score = RelevanceScore(
                            conceptual=0, structural=0, difficulty=0,
                            transferability=0, overall=0.0,
                            rationale=f"Parse error: {s}"
                        )
                    results.append({
                        **p,
                        "conceptual": score.conceptual,
                        "structural": score.structural,
                        "difficulty": score.difficulty,
                        "transferability": score.transferability,
                        "overall": score.overall,
                        "rationale": score.rationale,
                        "provider_used": "google",
                        "model_used": self.model_name,
                    })
                return results

            except Exception as e:
                last_err = e
                # exponential backoff on failure
                delay = min((self.base_delay * (2 ** attempt)) + random.random(), 15.0)
                time.sleep(delay)

        # if all retries fail → zeroed scores
        fallback = []
        for p in pairs:
            fallback.append({
                **p,
                "conceptual": 0, "structural": 0, "difficulty": 0,
                "transferability": 0, "overall": 0.0,
                "rationale": f"Error after retries: {last_err}",
                "provider_used": "google",
                "model_used": self.model_name,
            })
        return fallback


# --------------------------
# Utils
# --------------------------
def chunked(seq: List[Dict[str, Any]], size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# --------------------------
# Main
# --------------------------
def run_relevance(
    csv_path: str,
    out_csv: str,
    model: str = "gemini-2.5-flash",
    batch_size: int = 10,
    limit: int | None = None
) -> pd.DataFrame:

    df = pd.read_csv(csv_path)

    if limit is not None:
        df = df.head(limit)

    rows = df.to_dict(orient="records")
    client = LLMClient(model=model)
    results: List[Dict[str, Any]] = []

    for batch in tqdm(list(chunked(rows, batch_size)), desc="Evaluating relevance (batched)"):
        batch_results = client.evaluate_batch(batch)
        results.extend(batch_results)

    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    results_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"✅ Saved results to {out_csv} (rows={len(results_df)})")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batched relevance scoring with Gemini.")
    parser.add_argument("--csv_path", default="processed/normalized_all.csv", help="Input CSV")
    parser.add_argument("--out_csv", default="runs/google_results_all.csv", help="Output CSV")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name")
    parser.add_argument("--batch_size", type=int, default=10, help="Pairs per API call")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows (None = all)")
    args = parser.parse_args()

    run_relevance(
        csv_path=args.csv_path,
        out_csv=args.out_csv,
        model=args.model,
        batch_size=args.batch_size,
        limit=args.limit,
    )
