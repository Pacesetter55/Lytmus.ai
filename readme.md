# Similar Question Impact Evaluation

This project evaluates whether providing an LLM with **similar questions + their approaches** improves solution quality compared to solving independently.

## Project Structure
- **preprocess.py** → normalize raw JSON into a clean CSV
- **relevance.py** → score (question, similar_question) pairs with rubric
- **solution_builder.py** → generate baseline (no context) and augmented (with similar context) solutions
- **compare_solutions.py** → compare baseline vs augmented solutions
- **analyze_comparisons.py** → analyze results, deltas, and run A/B-style tests

## How it works
1. Preprocess dataset → `processed/normalized_all.csv`
2. Score relevance → `runs/google_results.csv`
3. Build solutions → `runs/solutions.csv`
4. Compare solutions → `runs/compare_solutions.csv`
5. Analyze → metrics + summaries in `runs/analysis/`

## Setup
```bash
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git
cd <YOUR_REPO>
python -m venv venv
venv\Scripts\activate    # Windows
pip install -r requirements.txt
