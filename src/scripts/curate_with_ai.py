#!/usr/bin/env python3
"""
AI-Assisted Dataset Curation Tool

Proces:
1. Losuje pytania z datasetu (po kolei)
2. AI ocenia czy pytanie jest OK (jasne, jednoznaczne, ma kontekst w odpowiedzi)
3. Jeśli AI mówi OK → pytam Cię o finalną akceptację
4. Jeśli AI mówi NIE OK → pokazuję ale sugeruję odrzucenie
5. Zbiera 100 zatwierdzonych pytań dla każdego datasetu

Usage:
    python3 curate_with_ai.py

Output:
    - curated_poquad_100.json - 100 zatwierdzonych pytań PoQuAD
    - curated_polqa_100.json - 100 zatwierdzonych pytań PolQA
"""

import sys

sys.path.append("..")

from dataset.polqa_dataset_getter import PolqaDatasetGetter
from dataset.poquad_dataset_getter import PoquadDatasetGetter
from common.names import DATASET_SEED
import mlx_lm
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

POQUAD_PROGRESS = os.path.join(OUTPUT_DIR, "poquad_ai_curation_progress.json")
POLQA_PROGRESS = os.path.join(OUTPUT_DIR, "polqa_ai_curation_progress.json")

POQUAD_OUTPUT = os.path.join(OUTPUT_DIR, "curated_poquad_100.json")
POLQA_OUTPUT = os.path.join(OUTPUT_DIR, "curated_polqa_100.json")

# AI Model - używamy najmniejszego modelu do szybkiej oceny (ścieżka absolutna)
AI_MODEL = "../../models/PLLuM-8B-instruct-q4"


def load_ai_model():
    """Load AI model for question evaluation"""
    print_colored("🤖 Loading AI model (PLLuM-8B)...", Colors.CYAN)

    # Konwertuj do ścieżki absolutnej
    model_path = os.path.abspath(AI_MODEL)
    if not os.path.exists(model_path):
        print_colored(f"❌ Model not found at: {model_path}", Colors.RED)
        print_colored("Available models:", Colors.YELLOW)
        models_dir = os.path.abspath("../../models")
        if os.path.exists(models_dir):
            for m in os.listdir(models_dir):
                print(f"  - {m}")
        raise FileNotFoundError(f"Model not found: {model_path}")

    print_colored(f"   Path: {model_path}", Colors.CYAN)
    model, tokenizer = mlx_lm.load(model_path)
    print_colored("✅ AI model loaded", Colors.GREEN)
    return model, tokenizer


def ai_evaluate_question(
    model, tokenizer, question: str, answers: list, context: str = None
) -> tuple:
    """
    AI evaluates if question is good for evaluation.

    Returns: (is_good: bool, reason: str, confidence: float)
    """

    # Prepare prompt
    answers_str = ", ".join(answers) if isinstance(answers, list) else answers

    prompt = f"""Oceń czy to pytanie nadaje się do ewaluacji systemu RAG (Retrieval-Augmented Generation).

Pytanie: {question}
Poprawna odpowiedź: {answers_str}

KRYTERIA ZŁEGO pytania (ODRZUĆ jeśli spełnia którekolwiek):
❌ Używa zaimków bez wyjaśnienia ("Gdzie on pojechał?", "Co ona napisała?", "W jakim roku to się stało?")
❌ Odnosi się do nieokreślonej osoby/rzeczy ("Do jakiej epoki się odwołuje?", "Jakie teksty napisał?")
❌ Wymaga kontekstu z poprzedniego pytania ("A w tym samym roku?", "Co jeszcze zrobił?", "Gdzie jeszcze?")
❌ Zbyt ogólne ("Z jakiej partii?", "Kiedy?", "Gdzie?")
❌ Pytanie tak/nie bez kontekstu ("Czy był tam?", "Czy to prawda?")
❌ Niejasny podmiot ("Kto to zrobił?", "Co to było?")

KRYTERIA DOBREGO pytania (ZAAKCEPTUJ tylko jeśli spełnia WSZYSTKIE):
✅ Pytanie zawiera pełną nazwę osoby/miejsca/rzeczy ("Jaką nagrodę otrzymał Jan Kowalski?")
✅ Pytanie jest kompletne i samodzielne - można je zrozumieć bez dodatkowego kontekstu
✅ Odpowiedź jest konkretna i jednoznaczna (imię, data, miejsce, liczba, nazwa)
✅ Pytanie ma sens dla kogoś kto widzi je pierwszy raz

PRZYKŁADY:
❌ ZŁE: "Do jakiej epoki literackiej odwołuje się w swoich tekstach Rymkiewicz?" (niejasne - w JAKICH tekstach?)
❌ ZŁE: "Z jakiej partii kandydował w 2014 r.?" (nie wiadomo KTO kandydował)
✅ DOBRE: "Do jakiej epoki literackiej odwołuje się Jarosław Marek Rymkiewicz w swoich wierszach?"
✅ DOBRE: "Z jakiej partii kandydował Adam Kowalski w wyborach do sejmu w 2014 r.?"

Oceń pytanie surowo. W razie wątpliwości - ODRZUĆ.

OCENA (DOBRE/ZŁE):"""

    # Generate AI response
    try:
        response = mlx_lm.generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=150,
        )

        response = response.strip().upper()

        # Parse response
        if "DOBRE" in response[:50]:
            # Extract reason
            reason_start = response.find("DOBRE") + 5
            reason = response[reason_start:].strip()
            if not reason or len(reason) < 10:
                reason = "Pytanie spełnia kryteria dobrego pytania ewaluacyjnego"
            return True, reason[:200], 0.8
        else:
            # Extract reason
            reason_start = response.find("ZŁE") + 3
            reason = response[reason_start:].strip()
            if not reason or len(reason) < 10:
                reason = "Pytanie nie spełnia kryteriów dobrego pytania ewaluacyjnego"
            return False, reason[:200], 0.8

    except Exception as e:
        print_colored(f"⚠️  AI evaluation error: {e}", Colors.YELLOW)
        # Fallback: neutral evaluation
        return True, "AI error - manual review needed", 0.0


def load_progress(progress_file):
    """Load curation progress"""
    if os.path.exists(progress_file):
        with open(progress_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "approved": [],
        "rejected": [],
        "ai_rejected": [],
        "current_index": 0,
        "questions": [],
    }


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


def curate_dataset_with_ai(
    model,
    tokenizer,
    dataset_name,
    dataset,
    progress_file,
    output_file,
    target_count=100,
):
    """Curate dataset with AI pre-filtering"""

    print_colored(f"\n{'='*80}", Colors.HEADER)
    print_colored(f"AI-ASSISTED CURATION: {dataset_name.upper()}", Colors.HEADER)
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
    ai_rejected_count = len(progress["ai_rejected"])

    print(f"Progress:")
    print_colored(f"  ✅ Approved: {approved_count}/{target_count}", Colors.GREEN)
    print_colored(f"  ❌ Rejected by you: {rejected_count}", Colors.RED)
    print_colored(f"  🤖 Rejected by AI: {ai_rejected_count}", Colors.YELLOW)
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
    print_colored("Process:", Colors.BOLD)
    print("  1. AI evaluates question quality")
    print("  2. If AI approves → You make final decision")
    print("  3. If AI rejects → You can override or skip")
    print()
    print_colored("Commands:", Colors.BOLD)
    print("  y/yes     - Approve (add to final dataset)")
    print("  n/no      - Reject")
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
            if (
                i in progress["approved"]
                or i in progress["rejected"]
                or i in progress["ai_rejected"]
            ):
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

            # AI EVALUATION
            print_colored(f"\n🤖 AI is evaluating...", Colors.CYAN)
            ai_good, ai_reason, ai_confidence = ai_evaluate_question(
                model, tokenizer, q["question"], q["answers"], q.get("context")
            )

            if ai_good:
                print_colored(f"🤖 AI: ✅ GOOD question", Colors.GREEN)
                print_colored(f"   Reason: {ai_reason}", Colors.GREEN)
            else:
                print_colored(f"🤖 AI: ❌ POOR question", Colors.RED)
                print_colored(f"   Reason: {ai_reason}", Colors.RED)

            # Show context preview if available
            if q.get("context"):
                context_preview = q["context"][:200]
                print(f"\n📄 Context (preview): {context_preview}...")

            # USER DECISION
            while True:
                if ai_good:
                    prompt = "\n👉 Do you APPROVE? (y/n/skip/context/quit): "
                else:
                    prompt = "\n👉 AI suggests REJECT. Override? (y=approve anyway, n=reject, skip/context/quit): "

                user_input = input(prompt).strip().lower()

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

                if user_input in ["y", "yes", "ok"]:
                    progress["approved"].append(i)
                    progress["current_index"] = i + 1
                    save_progress(progress, progress_file)
                    print_colored(
                        f"✅ APPROVED ({len(progress['approved'])}/{target_count})",
                        Colors.GREEN,
                    )
                    break

                if user_input in ["n", "no"]:
                    if ai_good:
                        # User rejected AI-approved question
                        progress["rejected"].append(i)
                    else:
                        # User confirmed AI rejection
                        progress["ai_rejected"].append(i)

                    progress["current_index"] = i + 1
                    save_progress(progress, progress_file)
                    print_colored("❌ Rejected", Colors.RED)
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
    print_colored("AI-ASSISTED DATASET CURATION", Colors.HEADER)
    print_colored("=" * 80 + "\n", Colors.HEADER)

    print(
        "This tool uses AI to pre-filter questions before you make the final decision."
    )
    print("AI evaluates question quality, then you approve/reject.\n")

    # Load AI model
    model, tokenizer = load_ai_model()

    # Load datasets - FULL test sets
    print_colored("\nLoading FULL datasets...", Colors.CYAN)
    poquad_getter = PoquadDatasetGetter()
    polqa_getter = PolqaDatasetGetter()

    # Get ALL test questions
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
        curate_dataset_with_ai(
            model, tokenizer, "PoQuAD", poquad_dataset, POQUAD_PROGRESS, POQUAD_OUTPUT
        )
    elif choice == "2":
        curate_dataset_with_ai(
            model, tokenizer, "PolQA", polqa_dataset, POLQA_PROGRESS, POLQA_OUTPUT
        )
    elif choice == "3":
        curate_dataset_with_ai(
            model, tokenizer, "PoQuAD", poquad_dataset, POQUAD_PROGRESS, POQUAD_OUTPUT
        )
        curate_dataset_with_ai(
            model, tokenizer, "PolQA", polqa_dataset, POLQA_PROGRESS, POLQA_OUTPUT
        )
    else:
        print_colored("❌ Invalid choice", Colors.RED)

    print_colored("\n" + "=" * 80, Colors.HEADER)
    print_colored("CURATION COMPLETE", Colors.HEADER)
    print_colored("=" * 80 + "\n", Colors.HEADER)
