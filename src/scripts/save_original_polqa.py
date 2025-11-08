#!/usr/bin/env python3
"""
Save the original PolQA 100 questions as curated dataset.

This ensures consistency with previous evaluations while allowing
fresh curation for PoQuAD.
"""

import sys

sys.path.append("..")

from dataset.polqa_dataset_getter import PolqaDatasetGetter
from common.names import DATASET_SEED
import json
import os

# Output
OUTPUT_DIR = "../../output/curated_datasets/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
POLQA_OUTPUT = os.path.join(OUTPUT_DIR, "curated_polqa_100.json")

print("Extracting original PolQA 100 questions...")

# Get the exact same 100 questions used in previous evaluations
polqa_getter = PolqaDatasetGetter()
polqa_dataset = polqa_getter.get_random_n_test(500, DATASET_SEED)[:100]

# Convert to curated format
curated_questions = [
    {
        "question": entry.question,
        "answers": (
            entry.answers if isinstance(entry.answers, list) else [entry.answers]
        ),
        "passage_id": entry.passage_id,
        "context": entry.context if hasattr(entry, "context") else None,
    }
    for entry in polqa_dataset
]

# Save
with open(POLQA_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(curated_questions, f, indent=2, ensure_ascii=False)

print(f"✅ Saved {len(curated_questions)} PolQA questions to: {POLQA_OUTPUT}")
print(
    f"   These are the SAME questions used in previous evaluations (seed={DATASET_SEED})"
)
print(f"\nNow curate PoQuAD using: python3 curate_with_ai.py")
