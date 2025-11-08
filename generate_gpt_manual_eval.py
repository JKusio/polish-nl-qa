#!/usr/bin/env python3
"""
Generate GPT manual_eval files - MINIMAL VERSION
Just load GPT answers from cache and save to manual_eval files
"""
import sys

sys.path.insert(0, "src")

import json
import csv
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance
from cache.cache import Cache
from common.utils import replace_slash_with_dash
from repository.qdrant_openai_repository import QdrantOpenAIRepository
from retrievers.qdrant_retriever import QdrantRetriever
from generators.openai_generator import OpenAIGenerator
from dataset.curated_dataset_getter import CuratedDatasetGetter

# Initialize
qdrant_client = QdrantClient(host="localhost", port=6333, check_compatibility=False)
cache = Cache()

# Load datasets
poquad_dataset = CuratedDatasetGetter.get_curated_poquad()
polqa_dataset = CuratedDatasetGetter.get_curated_polqa()

print("=" * 80)
print("LOADING GPT ANSWERS TO CACHE")
print("=" * 80)

# Load OpenAI answers to cache
datasets_keys = [
    "clarin-pl-poquad-1000",
    "clarin-pl-poquad-100000",
    "clarin-pl-poquad-2000",
    "clarin-pl-poquad-500",
    "ipipan-polqa-1000",
    "ipipan-polqa-100000",
    "ipipan-polqa-2000",
    "ipipan-polqa-500",
]

loaded_count = 0
for dataset_key in datasets_keys:
    filename = replace_slash_with_dash(f"gpt-4o-mini_{dataset_key}.jsonl")
    filepath = f"openai_batches/{filename}"

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                custom_id = data["custom_id"]
                answer = data["response"]["body"]["choices"][0]["message"]["content"]
                cache.set(custom_id, answer)
                loaded_count += 1
        print(f"✓ Loaded {filename}")
    except FileNotFoundError:
        print(f"✗ File not found: {filename}")

print(f"\n✅ Loaded {loaded_count} GPT answers to cache\n")


def clean_text_for_csv(text):
    if text is None:
        return ""
    cleaned = str(text).replace("\n", " ").replace("\r", " ")
    cleaned = " ".join(cleaned.split())
    return cleaned


def save_manual_eval(retriever, retriever_name, dataset, dataset_name, n):
    """Generate and save manual_eval file"""
    generator = OpenAIGenerator(cache)
    manual_rows = []

    print(f"  Processing {dataset_name} n={n}...")

    for i, entry in enumerate(
        tqdm(dataset, desc=f"  {dataset_name} n={n}", leave=False), 1
    ):
        question = entry.question
        correct_answers = entry.answers

        # Get retrieval results
        retriever_result = retriever.get_relevant_passages(question)
        passages = [passage for (passage, _) in retriever_result.passages]
        top_n_passages = passages[:n]

        # Generate answer from cache
        answer = generator.generate_answer(question, top_n_passages)

        # Prepare row
        question_id = f"{dataset_name}_q{i}"
        question_clean = clean_text_for_csv(question)
        answer_clean = clean_text_for_csv(answer)

        if isinstance(correct_answers, list):
            correct_answer_text = " | ".join(
                [clean_text_for_csv(ans) for ans in correct_answers]
            )
        else:
            correct_answer_text = clean_text_for_csv(str(correct_answers))

        manual_rows.append(
            [
                question_clean,
                question_id,
                "",  # hasCorrectPassages - will fill manually if needed
                "",  # ragas scores - skip for now
                "",
                "",
                "",
                "",
                answer_clean,
                correct_answer_text,
                "",  # manual_result
            ]
        )

    # Save file
    safe_retriever = retriever_name.replace("/", "_").replace("-", "_")[:50]
    filename = (
        f"ragas_v2_{dataset_name}_openai_{safe_retriever}_gpt_4o_mini_INST_n{n}.csv"
    )
    filepath = f"output/ragas_v2/manual_eval/{filename}"

    with open(filepath, mode="w", newline="", encoding="utf-8") as file:
        file.write(f"# RETRIEVER: {clean_text_for_csv(retriever_name)}\n")
        file.write(f"# GENERATOR: gpt-4o-mini\n")
        file.write(f"# TYPE: INST\n")
        file.write(f"# DATASET: {dataset_name}\n")
        file.write(f"# TOP_N: {n}\n")
        file.write(f"# CACHE_VERSION: openai\n")
        file.write(f"# RAGAS_VERSION: v2\n")
        file.write("\n")

        writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        writer.writerow(
            [
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
        )
        writer.writerows(manual_rows)

    print(f"  ✓ Saved: {filename}")


print("=" * 80)
print("GENERATING GPT MANUAL_EVAL FILES")
print("=" * 80)

# PoQuAD OpenAI
poquad_configs = [
    (
        "text-embedding-3-large-Cosine-clarin-pl-poquad-500",
        Distance.COSINE,
        "clarin-pl-poquad-500",
    ),
    (
        "text-embedding-3-large-Euclid-clarin-pl-poquad-2000",
        Distance.EUCLID,
        "clarin-pl-poquad-2000",
    ),
]

print("\n### POQUAD OPENAI ###\n")
for retriever_name, distance, dataset_key in poquad_configs:
    print(f"{retriever_name}:")
    repository = QdrantOpenAIRepository.get_repository(
        qdrant_client, "text-embedding-3-large", distance, cache
    )
    retriever = QdrantRetriever(repository, dataset_key)

    for n in [1, 5]:
        save_manual_eval(retriever, retriever_name, poquad_dataset, "poquad_openai", n)

# PolQA OpenAI
polqa_configs = [
    (
        "text-embedding-3-large-Euclid-ipipan-polqa-2000",
        Distance.EUCLID,
        "ipipan-polqa-2000",
    ),
    (
        "text-embedding-3-large-Cosine-ipipan-polqa-500",
        Distance.COSINE,
        "ipipan-polqa-500",
    ),
]

print("\n### POLQA OPENAI ###\n")
for retriever_name, distance, dataset_key in polqa_configs:
    print(f"{retriever_name}:")
    repository = QdrantOpenAIRepository.get_repository(
        qdrant_client, "text-embedding-3-large", distance, cache
    )
    retriever = QdrantRetriever(repository, dataset_key)

    for n in [1, 5]:
        save_manual_eval(retriever, retriever_name, polqa_dataset, "polqa_openai", n)

print("\n" + "=" * 80)
print("✅ DONE! All GPT manual_eval files generated with answers from cache")
print("=" * 80)
