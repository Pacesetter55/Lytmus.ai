import json
import os
import pandas as pd

# Paths
raw_path = "data/similar_question_data.json"
out_dir = "processed"
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, "normalized_all.csv")

# Load full dataset
with open(raw_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

def normalize_to_rows(entry):
    """Flatten one record into multiple row dictionaries (one per similar question)."""
    base = {
        "id": entry.get("question_id", ""),
        "subject": entry.get("subject", "Unknown"),
        "question": entry.get("question_text", ""),
    }
    rows = []
    for sim in entry.get("similar_questions", []):
        row = {
            **base,
            "similar_question": sim.get("similar_question_text", ""),
            "similarity_score": sim.get("similarity_score", None),
            "approach_summary": sim.get("summarized_solution_approach", None),
        }
        rows.append(row)
    return rows

# Normalize → flat list of rows
normalized_rows = []
for r in raw_data:
    normalized_rows.extend(normalize_to_rows(r))

# Convert to DataFrame
df = pd.DataFrame(normalized_rows)

# Save CSV
df.to_csv(csv_path, index=False, encoding="utf-8")

print(f"✅ Normalized {len(df)} rows → saved to {csv_path}")
print(df.head())
