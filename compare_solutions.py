# compare_solutions.py — Compare baseline vs augmented solutions with Gemini 2.5 Flash (batched, CSV I/O)

import os
import json
import time
import random
import argparse
from typing import List, Dict, Any

import pandas as pd
import google.generativeai as genai
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from tqdm import tqdm

# --------------------------
# Environment / API key
# --------------------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not set in environment or .env")
genai.configure(api_key=API_KEY)


# --------------------------
# Schemas
# --------------------------
class RubricScore(BaseModel):
    correctness: int = Field(..., ge=0, le=4)
    completeness: int = Field(..., ge=0, le=4)
    clarity: int = Field(..., ge=0, le=4)
    conciseness: int = Field(..., ge=0, le=4)
    overall: float = Field(..., ge=0, le=4)
    rationale: str


class PairEval(BaseModel):
    baseline: RubricScore
    augmented: RubricScore


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
    """
    Expect a JSON ARRAY. Handles markdown code fences and extra chatter.
    Returns a list[dict].
    """
    t = _strip_code_fences(text)
    # fast path
    try:
        data = json.loads(t)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # fallback: slice the first [...] block
    start, end = t.find("["), t.rfind("]")
    if start != -1 and end != -1 and end > start:
        data = json.loads(t[start : end + 1])
        if isinstance(data, list):
            return data

    raise ValueError("Model did not return a valid JSON array.")


# --------------------------
# LLM client (Google only, batched)
# --------------------------
class Evaluator:
    def __init__(self, model: str = "gemini-2.5-flash", max_retries: int = 5, base_delay: float = 2.0):
        self.model_name = model
        self.client = genai.GenerativeModel(model)
        self.max_retries = max_retries
        self.base_delay = base_delay

    def evaluate_batch(self, rows: List[Dict[str, Any]]) -> List[PairEval]:
        """
        rows: list of dicts with keys:
          id, subject, question, baseline_solution, augmented_solution
        Returns: list[PairEval] in same order/length.
        """
        header = (
            "You are a strict grader. For EACH item, score the BASELINE and AUGMENTED solutions to the same QUESTION.\n"
            "Rubric fields (0–4 unless noted): correctness, completeness, clarity, conciseness, overall (0–4 float), rationale (string).\n"
            "Return ONLY a JSON ARRAY with the SAME number of items and SAME ORDER as provided.\n"
            "Each item MUST have the shape:\n"
            '{ "baseline": {"correctness":0,"completeness":0,"clarity":0,"conciseness":0,"overall":0.0,"rationale":"..."},\n'
            '  "augmented": {"correctness":0,"completeness":0,"clarity":0,"conciseness":0,"overall":0.0,"rationale":"..."} }\n\n'
        )

        body_parts = []
        for i, r in enumerate(rows, start=1):
            body_parts.append(
                f"ITEM {i}:\nQUESTION:\n{r['question']}\n\nBASELINE SOLUTION:\n{r['baseline_solution']}\n\nAUGMENTED SOLUTION:\n{r['augmented_solution']}\n"
            )
        prompt = header + "\n".join(body_parts)

        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.generate_content(prompt)
                items = _parse_json_list_strict(resp.text)

                # pad/truncate to match length
                if len(items) < len(rows):
                    items += [
                        {
                            "baseline": {
                                "correctness": 0, "completeness": 0, "clarity": 0, "conciseness": 0, "overall": 0.0,
                                "rationale": "Model returned fewer items than requested."
                            },
                            "augmented": {
                                "correctness": 0, "completeness": 0, "clarity": 0, "conciseness": 0, "overall": 0.0,
                                "rationale": "Model returned fewer items than requested."
                            }
                        }
                        for _ in range(len(rows) - len(items))
                    ]
                elif len(items) > len(rows):
                    items = items[:len(rows)]

                parsed: List[PairEval] = []
                for obj in items:
                    try:
                        parsed.append(PairEval(**obj))
                    except ValidationError:
                        # salvage a malformed item with zeros
                        parsed.append(
                            PairEval(
                                baseline=RubricScore(
                                    correctness=0, completeness=0, clarity=0, conciseness=0, overall=0.0,
                                    rationale=f"Parse error: {obj}"
                                ),
                                augmented=RubricScore(
                                    correctness=0, completeness=0, clarity=0, conciseness=0, overall=0.0,
                                    rationale=f"Parse error: {obj}"
                                )
                            )
                        )
                return parsed

            except Exception as e:
                last_err = e
                delay = min((self.base_delay * (2 ** attempt)) + random.random(), 15.0)
                time.sleep(delay)

        # If all retries fail, return zeroed scores
        return [
            PairEval(
                baseline=RubricScore(correctness=0, completeness=0, clarity=0, conciseness=0, overall=0.0,
                                     rationale=f"Error after retries: {last_err}"),
                augmented=RubricScore(correctness=0, completeness=0, clarity=0, conciseness=0, overall=0.0,
                                      rationale=f"Error after retries: {last_err}")
            )
            for _ in rows
        ]


# --------------------------
# Utils
# --------------------------
def chunked(seq: List[Dict[str, Any]], size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# --------------------------
# Main
# --------------------------
def compare_solutions(
    solutions_csv: str,
    out_csv: str,
    model: str = "gemini-2.5-flash",
    batch_size: int = 8,
    limit: int | None = None
) -> pd.DataFrame:

    df = pd.read_csv(solutions_csv)

    # Expect columns from solution_builder.py
    needed = {"id", "subject", "question", "baseline_solution", "augmented_solution"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    # If you used --limit in solution_builder, you can limit again here for quick tests
    if limit is not None:
        # limit on unique questions
        uniq_ids = df["id"].drop_duplicates().head(limit)
        df = df[df["id"].isin(uniq_ids)]

    rows = df[["id", "subject", "question", "baseline_solution", "augmented_solution"]].to_dict(orient="records")

    evaluator = Evaluator(model=model)
    results_rows: List[Dict[str, Any]] = []

    for batch in tqdm(list(chunked(rows, batch_size)), desc="Scoring solutions (batched)"):
        evals = evaluator.evaluate_batch(batch)
        # Build output rows with both sets + deltas
        for r, e in zip(batch, evals):
            row = {
                "id": r["id"],
                "subject": r["subject"],
                "question": r["question"],
                # baseline
                "baseline_correctness": e.baseline.correctness,
                "baseline_completeness": e.baseline.completeness,
                "baseline_clarity": e.baseline.clarity,
                "baseline_conciseness": e.baseline.conciseness,
                "baseline_overall": e.baseline.overall,
                "baseline_rationale": e.baseline.rationale,
                # augmented
                "aug_correctness": e.augmented.correctness,
                "aug_completeness": e.augmented.completeness,
                "aug_clarity": e.augmented.clarity,
                "aug_conciseness": e.augmented.conciseness,
                "aug_overall": e.augmented.overall,
                "aug_rationale": e.augmented.rationale,
            }
            # deltas (aug - base)
            row["delta_correctness"] = row["aug_correctness"] - row["baseline_correctness"]
            row["delta_completeness"] = row["aug_completeness"] - row["baseline_completeness"]
            row["delta_clarity"] = row["aug_clarity"] - row["baseline_clarity"]
            row["delta_conciseness"] = row["aug_conciseness"] - row["baseline_conciseness"]
            row["delta_overall"] = row["aug_overall"] - row["baseline_overall"]

            results_rows.append(row)

    out_df = pd.DataFrame(results_rows)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"✅ Saved comparison to {out_csv} (rows={len(out_df)})")
    return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare baseline vs augmented solutions with Gemini 2.5 Flash")
    parser.add_argument("--input", default="runs/solutions.csv", help="Input CSV from solution_builder.py")
    parser.add_argument("--output", default="runs/compare_solutions.csv", help="Output CSV with comparison scores")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name")
    parser.add_argument("--batch_size", type=int, default=8, help="Items per API call")
    parser.add_argument("--limit", type=int, default=None, help="Limit unique questions (for sanity checks)")
    args = parser.parse_args()

    compare_solutions(
        solutions_csv=args.input,
        out_csv=args.output,
        model=args.model,
        batch_size=args.batch_size,
        limit=args.limit,
    )
