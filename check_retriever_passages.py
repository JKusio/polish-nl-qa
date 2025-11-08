#!/usr/bin/env python3
"""
Generate a column of TRUE/FALSE values showing if correct passage is in top-n results.
Easy to paste into Google Sheets.

Usage:
    python check_retriever_passages.py -d poquad -r basic_index -s 500 -n 5
    python check_retriever_passages.py -d poquad -r openai-euclid -s 2000 -n 1
    python check_retriever_passages.py -d poquad -r hybrid --es-index morfologik_index --qdrant-model intfloat/multilingual-e5-large --qdrant-distance cosine --alpha 0.5 --reranker sdadas/polish-reranker-large-ranknet -s 100000 -n 5
"""
import sys

sys.path.insert(0, "src")

import argparse
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from elasticsearch import Elasticsearch
from cache.cache import Cache
from repository.qdrant_repository import QdrantRepository
from repository.qdrant_openai_repository import QdrantOpenAIRepository
from repository.es_repository import ESRepository
from retrievers.qdrant_retriever import QdrantRetriever
from retrievers.es_retriever import ESRetriever
from retrievers.hybrid_retriever import HybridRetriever
from vectorizer.hf_vectorizer import HFVectorizer
from rerankers.hf_reranker import HFReranker
from common.utils import get_dataset_key
from common.models_dimensions import MODEL_DIMENSIONS_MAP
from common.names import OPENAI_EMBEDDING_MODEL_NAMES
import json


def main():
    parser = argparse.ArgumentParser(
        description="Check if correct passages are retrieved"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        choices=["polqa", "poquad"],
        help="Dataset: polqa lub poquad",
    )
    parser.add_argument(
        "-n", "--top", type=int, default=5, help="Liczba pasaży (default: 5)"
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=1000,
        choices=[500, 1000, 2000, 100000],
        help="Rozmiar datasetu: 500, 1000, 2000 lub 100000 (default: 1000)",
    )
    parser.add_argument(
        "-r",
        "--retriever",
        required=True,
        choices=[
            "es-basic",
            "es-polish",
            "es-morfologik",
            "qdrant-cosine",
            "qdrant-euclid",
            "openai-cosine",
            "openai-euclid",
            "hybrid",
        ],
        help="Retriever type",
    )
    # Hybrid retriever specific arguments
    parser.add_argument(
        "--es-index", type=str, help="ES index name for hybrid retriever"
    )
    parser.add_argument(
        "--qdrant-model", type=str, help="Qdrant model for hybrid retriever"
    )
    parser.add_argument(
        "--qdrant-distance",
        type=str,
        choices=["cosine", "euclid"],
        help="Qdrant distance metric",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for ES in hybrid (default: 0.5)",
    )
    parser.add_argument("--reranker", type=str, help="Reranker model")

    args = parser.parse_args()

    # Load questions
    dataset_name = "ipipan/polqa" if args.dataset == "polqa" else "clarin-pl/poquad"
    curated_file = f"output/curated_datasets/curated_{args.dataset}_100.json"

    with open(curated_file, "r") as f:
        questions = json.load(f)

    # Create retriever
    cache = Cache()
    dataset_key = get_dataset_key(dataset_name, args.size)

    if args.retriever == "hybrid":
        # Hybrid retriever
        if not args.es_index or not args.qdrant_model or not args.qdrant_distance:
            print(
                "Error: hybrid retriever requires --es-index, --qdrant-model, and --qdrant-distance"
            )
            return

        # ES Repository
        es_client = Elasticsearch(hosts=["http://localhost:9200"])
        es_repository = ESRepository(es_client, args.es_index, cache)

        # Qdrant Repository
        qdrant_client = QdrantClient(host="localhost", port=6333)
        qdrant_model = args.qdrant_model
        distance = (
            Distance.COSINE if args.qdrant_distance == "cosine" else Distance.EUCLID
        )
        distance_str = "Cosine" if distance == Distance.COSINE else "Euclid"
        collection_name = f"{qdrant_model.replace('/', '-')}-{distance_str}"

        dimension = MODEL_DIMENSIONS_MAP.get(qdrant_model)
        if not dimension:
            print(f"Error: Unknown model {qdrant_model}")
            return

        vectors_config = VectorParams(size=dimension, distance=distance)
        vectorizer = HFVectorizer(qdrant_model, cache)
        qdrant_repository = QdrantRepository(
            qdrant_client,
            collection_name,
            qdrant_model,
            vectors_config,
            vectorizer,
            cache,
            "",
            "zapytanie: ",
        )

        # Reranker (optional)
        reranker = None
        if args.reranker:
            reranker = HFReranker(args.reranker, cache)

        retriever = HybridRetriever(
            es_repository,
            qdrant_repository,
            dataset_key,
            alpha=args.alpha,
            reranker=reranker,
        )

        reranker_str = f"-{args.reranker}" if args.reranker else ""
        retriever_name = f"{args.es_index}-{collection_name}-{dataset_key}-{args.alpha}{reranker_str}"

    elif args.retriever.startswith("es-"):
        # Elasticsearch
        if args.retriever == "es-basic":
            index_name = "basic_index"
        elif args.retriever == "es-polish":
            index_name = "polish_index"
        elif args.retriever == "es-morfologik":
            index_name = "morfologik_index"
        else:
            index_name = "basic_index"

        client = Elasticsearch(hosts=["http://localhost:9200"])
        repository = ESRepository(client, index_name, cache)
        retriever = ESRetriever(repository, dataset_key)
        retriever_name = f"{args.retriever} ({index_name})"

    elif args.retriever.startswith("openai-"):
        # OpenAI embeddings via Qdrant
        model_name = OPENAI_EMBEDDING_MODEL_NAMES[0]
        distance = (
            Distance.COSINE if args.retriever == "openai-cosine" else Distance.EUCLID
        )

        client = QdrantClient(host="localhost", port=6333)
        repository = QdrantOpenAIRepository.get_repository(
            client, model_name, distance, cache
        )
        retriever = QdrantRetriever(repository, dataset_key)
        retriever_name = f"{repository.collection_name}-{dataset_key}"

    else:
        # Qdrant
        model_name = "sdadas/mmlw-retrieval-roberta-large"
        distance = (
            Distance.COSINE if args.retriever == "qdrant-cosine" else Distance.EUCLID
        )
        distance_str = "Cosine" if distance == Distance.COSINE else "Euclid"
        collection_name = f"sdadas-mmlw-retrieval-roberta-large-{distance_str}"

        client = QdrantClient(host="localhost", port=6333)
        dimension = MODEL_DIMENSIONS_MAP[model_name]
        vectors_config = VectorParams(size=dimension, distance=distance)

        vectorizer = HFVectorizer(model_name, cache)
        repository = QdrantRepository(
            client,
            collection_name,
            model_name,
            vectors_config,
            vectorizer,
            cache,
            "",
            "zapytanie: ",
        )
        retriever = QdrantRetriever(repository, dataset_key)
        retriever_name = f"{collection_name}-{dataset_key}"

    # Header info (for user reference, not for pasting)
    print("=" * 80)
    print(f"RETRIEVER: {retriever_name}")
    print(f"DATASET: {args.dataset.upper()}")
    print(f"TOP N: {args.top}")
    print(f"TOTAL QUESTIONS: {len(questions)}")
    print("=" * 80)
    print("\nChecking passages...\n")

    # Process all questions and collect results
    results = []
    correct_count = 0

    for i, question_data in enumerate(questions, 1):
        question = question_data["question"]
        correct_id = question_data.get("passage_id")

        # Get retrieval results
        if args.retriever == "hybrid":
            result = retriever.get_relevant_passages(question)
            passages = [passage for (passage, _) in result.passages[: args.top]]
        else:
            result = retriever.get_relevant_passages(question, size=args.top)
            passages = [passage for (passage, _) in result.passages]

        # Check if correct passage is in top n
        retrieved_ids = [passage.id for passage in passages]
        has_correct = correct_id in retrieved_ids

        results.append(has_correct)
        if has_correct:
            correct_count += 1

    # Print summary
    print("=" * 80)
    print(f"SUMMARY: {correct_count}/{len(questions)} questions have correct passage")
    print(f"ACCURACY: {correct_count/len(questions)*100:.2f}%")
    print("=" * 80)
    print("\nPASTE THIS INTO GOOGLE SHEETS (one column):")
    print("-" * 80)

    # Print results in Google Sheets format
    for has_correct in results:
        print("TRUE" if has_correct else "FALSE")


if __name__ == "__main__":
    main()
