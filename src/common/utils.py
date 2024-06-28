import uuid
from common.names import (
    CHUNK_SIZES,
    DATASET_NAMES,
    DISTANCES,
    MODEL_NAMES,
    SEMANTIC_TYPES,
)
from common.passage import Passage
import hashlib


def get_passages_for_embedding(dataset):
    unique_contexts = set((row["title"], row["context"]) for row in dataset)

    return list(
        map(lambda row: Passage(generate_id(), row[1], row[0]), unique_contexts)
    )


def generate_id():
    return str(uuid.uuid4())


def replace_slash_with_dash(text: str):
    return text.replace("/", "-")


def get_qdrant_collection_name(
    dataset_name: str,
    model_name: str,
    chunking_strategy: str,
    chunking_size: int,
    distance: str,
):
    return f"{dataset_name}-{model_name}-{chunking_strategy}-{chunking_size}-{distance}".replace(
        "/", "-"
    )


def get_prompt_hash(model: str, prompt: str):
    hashed = hashlib.sha256((model + prompt).encode()).hexdigest()
    return "prompt:" + hashed


def get_es_query_hash(index_name: str, dataset_key: str, query: str):
    hashed = hashlib.sha256((index_name + dataset_key + query).encode()).hexdigest()
    return "query:" + hashed


def get_all_qdrant_collection_names():
    names = []
    for dataset_name in DATASET_NAMES:
        for model_name in MODEL_NAMES:
            for distance in DISTANCES:
                for chunk_size, _ in CHUNK_SIZES:
                    name = get_qdrant_collection_name(
                        dataset_name, model_name, "character", chunk_size, distance
                    )
                    names.append(name)

                for semantic_type in SEMANTIC_TYPES:
                    name = get_qdrant_collection_name(
                        dataset_name, model_name, semantic_type, 1.5, distance
                    )
                    names.append(name)

    return names
