import json
import os
from qdrant_client import QdrantClient
from cache.cache import Cache
from common.names import OPENAI_EMBEDDING_MODEL_NAMES
from common.utils import (
    get_all_openai_model_combinations,
    get_generator_hash,
    replace_slash_with_dash,
)
from dataset.curated_dataset_getter import CuratedDatasetGetter
from repository.qdrant_openai_repository import QdrantOpenAIRepository
from qdrant_client.models import Distance
from retrievers.qdrant_retriever import QdrantRetriever
from retrievers.retriever import Retriever


def main():
    qdrant_client = QdrantClient(host="localhost", port=6333)
    cache = Cache()

    get_batch_file(qdrant_client, cache)


def get_retriever(
    qdrantClient: QdrantClient, distance: Distance, cache: Cache, dataset_key: str
) -> Retriever:
    repository = QdrantOpenAIRepository.get_repository(
        qdrantClient, OPENAI_EMBEDDING_MODEL_NAMES[0], distance, cache
    )

    retriever = QdrantRetriever(repository, dataset_key)

    return retriever


def get_batch_file(qdrantClient: QdrantClient, cache: Cache):
    combinations = get_all_openai_model_combinations()

    # Use CURATED datasets instead of random
    poquad_dataset = CuratedDatasetGetter.get_curated_poquad()
    polqa_dataset = CuratedDatasetGetter.get_curated_polqa()

    # Create output directory if it doesn't exist
    os.makedirs("openai_batches", exist_ok=True)

    print(f"Using CURATED datasets:")
    print(f"  PoQuAD: {len(poquad_dataset)} questions")
    print(f"  PolQA: {len(polqa_dataset)} questions")

    custom_ids = set()

    for model, distance, dataset_key in combinations:
        retriever = get_retriever(qdrantClient, distance, cache, dataset_key)

        dataset = poquad_dataset if "poquad" in dataset_key else polqa_dataset
        dataset_name = "poquad_curated" if "poquad" in dataset_key else "polqa_curated"

        for entry in dataset:
            # Use "curated" in filename to distinguish from random datasets
            filename = replace_slash_with_dash(
                f"gpt-4o-mini_{dataset_name}_{dataset_key}.jsonl"
            )
            file_path = f"openai_batches/{filename}"

            result = retriever.get_relevant_passages(entry.question)

            ns = [1, 3, 5, 10]

            for n in ns:
                passages = [passage for (passage, _) in result.passages]
                top_n_passages = passages[:n]

                context = " ".join(
                    [passage.context for passage in top_n_passages]
                ).replace("\n", " ")

                hash_key = get_generator_hash(
                    entry.question, context, "instruction", "gpt-4o-mini"
                )

                if hash_key in custom_ids:
                    continue

                custom_ids.add(hash_key)

                prompt = f"""
                [INST]
                    Wygeneruj krótką odpowiedź na pytanie wyłącznie na podstawie poniższego kontekstu:
                    {context}

                    Pytanie: {entry.question}
                [/INST]
                """

                batch_request = {
                    "custom_id": f"{hash_key}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "user", "content": prompt.replace("\n", " ")}
                        ],
                        "max_tokens": 1000,
                    },
                }

                if os.path.exists(file_path):
                    # Open the file in append mode
                    with open(file_path, "a", encoding="utf-8") as f:
                        f.write(f"{json.dumps(batch_request, ensure_ascii=False)}\n")
                else:
                    # Open the file in write mode
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(f"{json.dumps(batch_request, ensure_ascii=False)}\n")

    print(f"\n✅ Generated {len(custom_ids)} unique batch requests")
    print(f"Files saved to: openai_batches/gpt-4o-mini_*_curated_*.jsonl")

    # List generated files
    print("\nGenerated files:")
    for model, distance, dataset_key in combinations:
        dataset_name = "poquad_curated" if "poquad" in dataset_key else "polqa_curated"
        filename = replace_slash_with_dash(
            f"gpt-4o-mini_{dataset_name}_{dataset_key}.jsonl"
        )
        filepath = f"openai_batches/{filename}"
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                line_count = sum(1 for _ in f)
            print(f"  {filename}: {line_count} requests")


if __name__ == "__main__":
    main()
