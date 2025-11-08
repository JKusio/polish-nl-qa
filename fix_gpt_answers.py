#!/usr/bin/env python3
"""
Fix GPT manual_eval files - add answers from detailed results
"""
import csv
import os

# Map from detailed results
detailed_file = "output/ragas_v2/ragas_v2_detailed.csv"

print("=" * 80)
print("FIXING GPT MANUAL_EVAL FILES - ADDING ANSWERS")
print("=" * 80)

# Load detailed results into memory
print("\n Loading detailed results...")
detailed_data = {}

with open(detailed_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["generator"] == "gpt-4o-mini":
            # Create key: dataset_retriever_n_question
            key = f"{row['dataset']}_{row['retriever']}_{row['n']}_{row['question']}"
            detailed_data[key] = row["answer"]

print(f"✅ Loaded {len(detailed_data)} GPT answers from detailed results")

# Process manual_eval files
manual_eval_dir = "output/ragas_v2/manual_eval/"
files_to_fix = [
    f
    for f in os.listdir(manual_eval_dir)
    if f.startswith("ragas_v2_") and "gpt_4o_mini" in f and f.endswith(".csv")
]

print(f"\nFound {len(files_to_fix)} GPT manual_eval files to fix\n")

fixed_count = 0
for filename in files_to_fix:
    filepath = os.path.join(manual_eval_dir, filename)

    # Parse filename to extract dataset, retriever, n
    # Format: ragas_v2_{dataset}_{retriever}_gpt_4o_mini_INST_n{n}.csv
    parts = (
        filename.replace("ragas_v2_", "")
        .replace("_gpt_4o_mini_INST_n", "|")
        .replace(".csv", "")
        .split("|")
    )
    if len(parts) != 2:
        print(f"✗ Skipped {filename} - unexpected format")
        continue

    dataset_retriever = parts[0]
    n = parts[1]

    # Determine dataset
    if "polqa_openai" in dataset_retriever:
        dataset = "polqa_openai"
        retriever = dataset_retriever.replace("polqa_openai_", "")
    elif "poquad_openai" in dataset_retriever:
        dataset = "poquad_openai"
        retriever = dataset_retriever.replace("poquad_openai_", "")
    else:
        print(f"✗ Skipped {filename} - unknown dataset")
        continue

    # Read file
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Separate metadata from CSV data
    metadata_lines = []
    csv_start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("#"):
            metadata_lines.append(line)
        else:
            csv_start_idx = i
            break

    # Parse CSV
    csv_content = "".join(lines[csv_start_idx:])
    reader = csv.DictReader(csv_content.splitlines())
    rows = list(reader)

    if not rows:
        print(f"✗ Skipped {filename} - no data rows")
        continue

    # Debug - show keys
    if fixed_count == 0:
        print(f"DEBUG: First file keys: {list(rows[0].keys())}")

    # Update answers
    updated_count = 0
    for row in rows:
        question = row.get("question", "")
        if not question:
            continue
        # Try to find answer in detailed data
        key = f"{dataset}_{retriever}_{n}_{question}"

        if key in detailed_data:
            row["answer"] = detailed_data[key]
            updated_count += 1
        else:
            # Try without underscores in retriever (normalize)
            key_normalized = f"{dataset}_{retriever.replace('_', '-')}_{n}_{question}"
            if key_normalized in detailed_data:
                row["answer"] = detailed_data[key_normalized]
                updated_count += 1

    # Write updated file
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        # Write metadata
        for line in metadata_lines:
            f.write(line)
        f.write("\n")

        # Write CSV
        fieldnames = rows[0].keys() if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ Fixed {filename} ({updated_count}/{len(rows)} answers added)")
    fixed_count += 1

print("\n" + "=" * 80)
print(f"✅ DONE! Fixed {fixed_count} files")
print("=" * 80)
