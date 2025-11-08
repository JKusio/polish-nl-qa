#!/usr/bin/env python3
"""
Manual Dataset Curation - NO AI, just you!

Losuje pytania i pokazuje Tobie do decyzji.
Szybko i bez marnowania mocy obliczeniowej na AI.

Usage:
    python3 curate_manual.py

Output:
    - curated_poquad_100.json - 100 zatwierdzonych pytań PoQuAD
    - curated_polqa_100.json - 100 zatwierdzonych pytań PolQA
"""

import sys

sys.path.append("..")

from dataset.polqa_dataset_getter import PolqaDatasetGetter
from dataset.poquad_dataset_getter import PoquadDatasetGetter
from common.names import DATASET_SEED
import json
import os
import random


# Colors
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    END = "\033[0m"
    BOLD = "\033[1m"


def print_colored(text, color):
    print(f"{color}{text}{Colors.END}")


# Config
OUTPUT_DIR = "../../output/curated_datasets/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

POQUAD_PROGRESS = os.path.join(OUTPUT_DIR, "poquad_manual_progress.json")
POLQA_PROGRESS = os.path.join(OUTPUT_DIR, "polqa_manual_progress.json")

POQUAD_OUTPUT = os.path.join(OUTPUT_DIR, "curated_poquad_100.json")
POLQA_OUTPUT = os.path.join(OUTPUT_DIR, "curated_polqa_100.json")


def load_progress(progress_file):
    """Load curation progress"""
    if os.path.exists(progress_file):
        with open(progress_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"approved": [], "rejected": [], "current_index": 0, "questions": []}


def save_progress(progress, progress_file):
    """Save curation progress"""
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def save_final_dataset(questions, output_file):
    """Save final curated dataset"""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    print_colored(
        f"\n✅ Saved {len(questions)} questions to: {output_file}", Colors.GREEN
    )


def curate_dataset_manual(
    dataset_name, dataset, progress_file, output_file, target_count=100
):
    """Curate dataset manually - no AI"""

    print_colored(f"\n{'='*80}", Colors.HEADER)
    print_colored(f"MANUAL CURATION: {dataset_name.upper()}", Colors.HEADER)
    print_colored(f"{'='*80}\n", Colors.HEADER)

    # Load progress
    progress = load_progress(progress_file)

    # Initialize questions pool if first run
    if not progress["questions"]:
        # Shuffle entire dataset
        shuffled = list(dataset)
        random.Random(DATASET_SEED).shuffle(shuffled)

        progress["questions"] = [
            {
                "question": entry.question,
                "answers": (
                    entry.answers
                    if isinstance(entry.answers, list)
                    else [entry.answers]
                ),
                "passage_id": entry.passage_id,
                "context": entry.context if hasattr(entry, "context") else None,
            }
            for entry in shuffled
        ]
        save_progress(progress, progress_file)

    approved_count = len(progress["approved"])
    rejected_count = len(progress["rejected"])
    total_reviewed = approved_count + rejected_count

    print(f"Progress:")
    print_colored(f"  ✅ Approved: {approved_count}/{target_count}", Colors.GREEN)
    print_colored(f"  ❌ Rejected: {rejected_count}", Colors.RED)
    print(f"  📊 Reviewed: {total_reviewed}/{len(progress['questions'])}")
    print()

    # Check if done
    if approved_count >= target_count:
        print_colored(
            f"✅ Already have {approved_count} approved questions!", Colors.GREEN
        )
        approved_questions = [
            progress["questions"][i] for i in progress["approved"][:target_count]
        ]
        save_final_dataset(approved_questions, output_file)
        return

    # Review questions
    questions = progress["questions"]
    start_idx = progress["current_index"]

    print_colored(
        f"Starting from question {start_idx + 1}/{len(questions)}\n", Colors.CYAN
    )
    print_colored("Commands:", Colors.BOLD)
    print("  y/yes     - Approve this question")
    print("  n/no      - Reject this question")
    print("  skip      - Skip for now")
    print("  quit      - Save and exit")
    print("  context   - Show full context")
    print()

    try:
        for i in range(start_idx, len(questions)):
            # Check if done
            if len(progress["approved"]) >= target_count:
                print_colored(
                    f"\n🎉 Reached target of {target_count} approved questions!",
                    Colors.GREEN,
                )
                break

            # Skip if already reviewed
            if i in progress["approved"] or i in progress["rejected"]:
                continue

            q = questions[i]

            print_colored(f"\n{'─'*80}", Colors.CYAN)
            print_colored(f"QUESTION {i + 1}/{len(questions)}", Colors.BOLD)
            print_colored(
                f"Progress: {len(progress['approved'])}/{target_count} approved",
                Colors.CYAN,
            )
            print_colored(f"{'─'*80}", Colors.CYAN)

            print(f"\n📝 Question: {q['question']}")
            print(f"\n✅ Correct answer(s):")
            for ans in q["answers"]:
                print(f"   - {ans}")

            # Show context preview if available
            if q.get("context"):
                context_preview = q["context"][:300]
                print(f"\n📄 Context (preview):")
                print_colored(f"{context_preview}...", Colors.YELLOW)

            # User input
            while True:
                user_input = (
                    input(f"\n👉 Approve this question? (y/n/skip/context/quit): ")
                    .strip()
                    .lower()
                )

                if user_input in ["q", "quit"]:
                    progress["current_index"] = i
                    save_progress(progress, progress_file)
                    print_colored(
                        "\n💾 Progress saved. Run again to continue.", Colors.GREEN
                    )
                    return

                if user_input in ["s", "skip"]:
                    print_colored("⏭️  Skipped", Colors.YELLOW)
                    break

                if user_input in ["c", "context", "show"]:
                    if q.get("context"):
                        print_colored(f"\n{'='*80}", Colors.BLUE)
                        print_colored("FULL CONTEXT:", Colors.BOLD)
                        print_colored(f"{'='*80}", Colors.BLUE)
                        print(q["context"])
                        print_colored(f"{'='*80}\n", Colors.BLUE)
                    else:
                        print_colored("❌ No context available", Colors.RED)
                    continue

                if user_input in ["y", "yes", "ok", "good"]:
                    progress["approved"].append(i)
                    progress["current_index"] = i + 1
                    save_progress(progress, progress_file)
                    print_colored(
                        f"✅ Approved ({len(progress['approved'])}/{target_count})",
                        Colors.GREEN,
                    )
                    break

                if user_input in ["n", "no", "bad", "reject"]:
                    progress["rejected"].append(i)
                    progress["current_index"] = i + 1
                    save_progress(progress, progress_file)
                    print_colored(
                        f"❌ Rejected (total rejected: {len(progress['rejected'])})",
                        Colors.RED,
                    )
                    break

                print_colored(
                    "❓ Unknown command. Use: y/n/skip/context/quit", Colors.YELLOW
                )

    except KeyboardInterrupt:
        print_colored("\n\n⚠️  Interrupted by user", Colors.YELLOW)
        progress["current_index"] = i
        save_progress(progress, progress_file)
        print_colored("💾 Progress saved. Run again to continue.", Colors.GREEN)
        return

    # Save final dataset if done
    if len(progress["approved"]) >= target_count:
        approved_questions = [questions[i] for i in progress["approved"][:target_count]]
        save_final_dataset(approved_questions, output_file)
    else:
        needed = target_count - len(progress["approved"])
        print_colored(
            f"\n⚠️  Need {needed} more questions to reach {target_count}", Colors.YELLOW
        )


# Main
if __name__ == "__main__":
    print_colored("\n" + "=" * 80, Colors.HEADER)
    print_colored("MANUAL DATASET CURATION (NO AI)", Colors.HEADER)
    print_colored("=" * 80 + "\n", Colors.HEADER)

    print("Simple and fast - just you reviewing questions!")
    print("No AI overhead, pure manual curation.\n")

    # Load datasets
    print_colored("Loading datasets...", Colors.CYAN)
    poquad_getter = PoquadDatasetGetter()
    polqa_getter = PolqaDatasetGetter()

    poquad_dataset = poquad_getter.get_test_dataset()
    polqa_dataset = polqa_getter.get_test_dataset()

    print(f"✅ Loaded {len(poquad_dataset)} PoQuAD questions (full test set)")
    print(f"✅ Loaded {len(polqa_dataset)} PolQA questions (full test set)\n")

    # Choose dataset
    print_colored("Which dataset to curate?", Colors.BOLD)
    print("  1 - PoQuAD")
    print("  2 - PolQA")
    print("  3 - Both (PoQuAD first, then PolQA)")
    print()

    choice = input("Enter choice (1/2/3): ").strip()

    if choice == "1":
        curate_dataset_manual("PoQuAD", poquad_dataset, POQUAD_PROGRESS, POQUAD_OUTPUT)
    elif choice == "2":
        curate_dataset_manual("PolQA", polqa_dataset, POLQA_PROGRESS, POLQA_OUTPUT)
    elif choice == "3":
        curate_dataset_manual("PoQuAD", poquad_dataset, POQUAD_PROGRESS, POQUAD_OUTPUT)
        curate_dataset_manual("PolQA", polqa_dataset, POLQA_PROGRESS, POLQA_OUTPUT)
    else:
        print_colored("❌ Invalid choice", Colors.RED)

    print_colored("\n" + "=" * 80, Colors.HEADER)
    print_colored("CURATION COMPLETE", Colors.HEADER)
    print_colored("=" * 80 + "\n", Colors.HEADER)
