#!/usr/bin/env python3
"""
Uniwersalny viewer - wybierz wszystko z CLI
"""
import sys, os

sys.path.insert(0, "src")

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
    import argparse

    parser = argparse.ArgumentParser(description="Uniwersalny viewer retrieverów")
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        choices=["polqa", "poquad"],
        help="Dataset: polqa lub poquad",
    )
    parser.add_argument(
        "-q", "--question", required=True, type=int, help="Numer pytania (1-100)"
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
        help="Retriever: es-basic, es-polish, es-morfologik, qdrant-cosine, qdrant-euclid, openai-cosine, openai-euclid, hybrid",
    )
    # Hybrid retriever specific arguments
    parser.add_argument(
        "--es-index",
        type=str,
        help="ES index name for hybrid retriever (e.g., morfologik_index)",
    )
    parser.add_argument(
        "--qdrant-model",
        type=str,
        help="Qdrant model for hybrid retriever (e.g., intfloat/multilingual-e5-large)",
    )
    parser.add_argument(
        "--qdrant-distance",
        type=str,
        choices=["cosine", "euclid"],
        help="Qdrant distance metric for hybrid",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for ES in hybrid retriever (default: 0.5)",
    )
    parser.add_argument(
        "--reranker",
        type=str,
        help="Reranker model (e.g., sdadas/polish-reranker-large-ranknet)",
    )

    args = parser.parse_args()

    # Load questions
    dataset_name = "ipipan/polqa" if args.dataset == "polqa" else "clarin-pl/poquad"
    curated_file = f"output/curated_datasets/curated_{args.dataset}_100.json"

    with open(curated_file, "r") as f:
        questions = json.load(f)

    if not (1 <= args.question <= len(questions)):
        print(f"Błąd: pytanie musi być 1-{len(questions)}")
        return

    question_data = questions[args.question - 1]
    question = question_data["question"]
    answers = question_data["answers"]
    correct_id = question_data.get("passage_id")

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
        model_name = OPENAI_EMBEDDING_MODEL_NAMES[0]  # text-embedding-3-large
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

    # Print actual retriever details
    print(f"\n{'=' * 100}")
    print(f"DATASET: {args.dataset.upper()}")
    print(f"RETRIEVER: {retriever_name}")
    print(f"PYTANIE #{args.question}: {question}")
    print(f"ODPOWIEDŹ: {', '.join(answers)}")
    print("=" * 100)

    # Get results
    print(f"\nSzukanie TOP {args.top} pasaży...\n")

    # Hybrid retriever doesn't take size parameter
    if args.retriever == "hybrid":
        result = retriever.get_relevant_passages(question)
        # Limit to requested number
        result.passages = result.passages[: args.top]
    else:
        result = retriever.get_relevant_passages(question, size=args.top)

    # Show passages
    for i, (passage, score) in enumerate(result.passages, 1):
        is_correct = "✓ POPRAWNY" if correct_id and passage.id == correct_id else ""
        print(f"{'#' * 100}")
        print(f"PASAŻ #{i} (Score: {score:.4f}) {is_correct}")
        print("#" * 100)
        print(f"Tytuł: {passage.title}")
        print(f"\n{passage.context}\n")
        print(f"ID: {passage.id}")
        print()

    # Summary
    found_at = None
    for i, (passage, _) in enumerate(result.passages, 1):
        if passage.id == correct_id:
            found_at = i
            break

    print("=" * 100)
    if found_at:
        print(f"✓✓✓ POPRAWNY PASAŻ NA POZYCJI {found_at} ✓✓✓")
    else:
        print(f"✗✗✗ POPRAWNY PASAŻ NIE W TOP-{args.top} ✗✗✗")
    print("=" * 100)


if __name__ == "__main__":
    main()
