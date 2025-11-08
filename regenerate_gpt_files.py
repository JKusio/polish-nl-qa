#!/usr/bin/env python3
"""
Regenerate GPT manual_eval files with correct answers
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
from common.names import DATASET_SEED
from repository.qdrant_openai_repository import QdrantOpenAIRepository
from retrievers.qdrant_retriever import QdrantRetriever
from generators.openai_generator import OpenAIGenerator
from dataset.polqa_dataset_getter import PolqaDatasetGetter
from dataset.poquad_dataset_getter import PoquadDatasetGetter
from dataset.curated_dataset_getter import CuratedDatasetGetter
from evaluation.ragas_evaulator_v2 import RAGASEvaluatorV2
from vectorizer.hf_vectorizer import HFVectorizer

# Initialize
qdrant_client = QdrantClient(host="localhost", port=6333)
cache = Cache()

# Initialize vectorizer and RAGAS V2
vectorizer = HFVectorizer("intfloat/multilingual-e5-large", cache)
ragas_evaluator = RAGASEvaluatorV2(
    reranker_model_name="sdadas/polish-reranker-large-ranknet",
    cache=cache,
    generator_model_name="models/PLLuM-12-B-instruct-q4",
    vectorizer=vectorizer,
)

# Load datasets (curated)
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
    """Clean text to be CSV-safe"""
    if text is None:
        return ""
    cleaned = str(text).replace("\n", " ").replace("\r", " ")
    cleaned = " ".join(cleaned.split())
    return cleaned


def evaluate_and_save(retriever, retriever_name, dataset, dataset_name, n):
    """Evaluate GPT and save manual_eval file"""
    generator = OpenAIGenerator(cache)
    manual_rows = []

    for i, entry in enumerate(tqdm(dataset, desc=f"{dataset_name} n={n}")):
        question = entry.question
        correct_passage_id = entry.passage_id
        correct_answers = entry.answers

        # Get retrieval results
        retriever_result = retriever.get_relevant_passages(question)
        passages = [passage for (passage, _) in retriever_result.passages]
        top_n_passages = passages[:n]

        # Generate answer (from cache)
        answer = generator.generate_answer(question, top_n_passages)

        # Check if correct passage is in top n
        retrieved_ids = [passage.id for passage in top_n_passages]
        has_correct_passages = str(correct_passage_id in retrieved_ids).upper()

        # Evaluate with RAGAS V2
        ragas_score = ragas_evaluator.ragas(
            retriever_result,
            correct_passage_id,
            answer,
            correct_answers=correct_answers,
        )
        faithfulness = ragas_evaluator.faithfulness(retriever_result, answer)
        answer_relevance = ragas_evaluator.answer_relevance(question, answer)
        answer_correctness = ragas_evaluator.answer_correctness(answer, correct_answers)
        context_recall = ragas_evaluator.context_recall(
            retriever_result, correct_passage_id
        )

        # Prepare row
        question_id = f"{dataset_name}_q{i+1}"
        question_text_clean = clean_text_for_csv(question)
        answer_clean = clean_text_for_csv(answer)

        if isinstance(correct_answers, list):
            correct_answer_text = " | ".join(
                [clean_text_for_csv(ans) for ans in correct_answers]
            )
        else:
            correct_answer_text = clean_text_for_csv(str(correct_answers))

        manual_rows.append(
            [
                question_text_clean,
                question_id,
                has_correct_passages,
                f"{ragas_score:.4f}",
                f"{faithfulness:.4f}",
                f"{answer_relevance:.4f}",
                f"{answer_correctness:.4f}",
                f"{context_recall:.4f}",
                answer_clean,
                correct_answer_text,
                "",
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
print("REGENERATING GPT MANUAL_EVAL FILES")
print("=" * 80)

# PoQuAD OpenAI retrievers
poquad_openai_configs = [
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

print("\n### POQUAD OPENAI ###")
for retriever_name, distance, dataset_key in poquad_openai_configs:
    print(f"\n{retriever_name}:")
    repository = QdrantOpenAIRepository.get_repository(
        qdrant_client, "text-embedding-3-large", distance, cache
    )
    retriever = QdrantRetriever(repository, dataset_key)

    for n in [1, 5]:
        evaluate_and_save(retriever, retriever_name, poquad_dataset, "poquad_openai", n)

# PolQA OpenAI retrievers
polqa_openai_configs = [
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

print("\n### POLQA OPENAI ###")
for retriever_name, distance, dataset_key in polqa_openai_configs:
    print(f"\n{retriever_name}:")
    repository = QdrantOpenAIRepository.get_repository(
        qdrant_client, "text-embedding-3-large", distance, cache
    )
    retriever = QdrantRetriever(repository, dataset_key)

    for n in [1, 5]:
        evaluate_and_save(retriever, retriever_name, polqa_dataset, "polqa_openai", n)

print("\n" + "=" * 80)
print("✅ DONE! All GPT manual_eval files regenerated with correct answers")
print("=" * 80)
