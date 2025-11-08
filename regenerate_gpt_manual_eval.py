#!/usr/bin/env python3
"""
Regenerate GPT manual_eval files from detailed results
"""
import csv
import os

# Map from detailed results
detailed_file = "output/ragas_v2/ragas_v2_detailed.csv"

print("=" * 80)
print("REGENERATING GPT MANUAL_EVAL FILES FROM DETAILED RESULTS")
print("=" * 80)

# Load detailed results
print("\nLoading detailed results...")
gpt_results = {}

with open(detailed_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["generator"] == "gpt-4o-mini":
            # Group by dataset, retriever, n
            dataset = row["dataset"]
            retriever = row["retriever"]
            n = row["n"]

            key = (dataset, retriever, n)
            if key not in gpt_results:
                gpt_results[key] = []

            gpt_results[key].append(row)

print(
    f"✅ Loaded {sum(len(v) for v in gpt_results.values())} GPT results in {len(gpt_results)} configurations\n"
)


# Helper function
def clean_text_for_csv(text):
    if text is None:
        return ""
    cleaned = str(text).replace("\n", " ").replace("\r", " ")
    cleaned = " ".join(cleaned.split())
    return cleaned


def create_safe_filename(dataset, retriever, n):
    """Create safe filename"""
    safe_retriever = retriever.replace("/", "_").replace("-", "_")
    if len(safe_retriever) > 50:
        import hashlib

        h = hashlib.md5(safe_retriever.encode()).hexdigest()[:8]
        safe_retriever = f"{safe_retriever[:40]}_{h}"

    return f"ragas_v2_{dataset}_{safe_retriever}_gpt_4o_mini_INST_n{n}.csv"


# Regenerate files
manual_eval_dir = "output/ragas_v2/manual_eval/"
os.makedirs(manual_eval_dir, exist_ok=True)

regenerated = 0
for (dataset, retriever, n), results in gpt_results.items():
    filename = create_safe_filename(dataset, retriever, n)
    filepath = os.path.join(manual_eval_dir, filename)

    # Prepare rows
    manual_rows = []
    for i, row in enumerate(results, 1):
        question_id = f"{dataset}_q{i}"

        # Check if correct passage in retrieved
        # We don't have this info in detailed, so leave empty or default
        has_correct = ""  # Will be filled later if needed

        manual_rows.append(
            {
                "question": clean_text_for_csv(row["question"]),
                "question_id": question_id,
                "hasCorrectPassages": has_correct,
                "ragas_v2_score": f"{float(row['ragas_v2']):.4f}",
                "faithfulness": f"{float(row['faithfulness']):.4f}",
                "answer_relevance": f"{float(row['answer_relevance']):.4f}",
                "answer_correctness": f"{float(row['answer_correctness']):.4f}",
                "context_recall": f"{float(row['context_recall']):.4f}",
                "answer": clean_text_for_csv(row["answer"]),
                "correct_answer": clean_text_for_csv(row["correct_answers"]),
                "manual_result": "",
            }
        )

    # Write file
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        # Write metadata
        f.write(f"# RETRIEVER: {clean_text_for_csv(retriever)}\n")
        f.write(f"# GENERATOR: gpt-4o-mini\n")
        f.write(f"# TYPE: INST\n")
        f.write(f"# DATASET: {dataset}\n")
        f.write(f"# TOP_N: {n}\n")
        f.write(f"# CACHE_VERSION: openai\n")
        f.write(f"# RAGAS_VERSION: v2\n")
        f.write("\n")

        # Write CSV
        fieldnames = [
            "question",
            "question_id",
            "hasCorrectPassages",
            "ragas_v2_score",
            "faithfulness",
            "answer_relevance",
            "answer_correctness",
            "context_recall",
            "answer",
            "correct_answer",
            "manual_result",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(manual_rows)

    print(f"✓ Regenerated {filename} ({len(manual_rows)} questions)")
    regenerated += 1

print("\n" + "=" * 80)
print(f"✅ DONE! Regenerated {regenerated} GPT manual_eval files")
print("=" * 80)
