import uuid
from common.names import (
    CHUNK_SIZES,
    DATASET_NAMES,
    DISTANCES,
    INDEX_NAMES,
    MODEL_NAMES,
    OPENAI_EMBEDDING_MODEL_NAMES,
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
    model_name: str,
    distance: str,
):
    return replace_slash_with_dash(f"{model_name}-{distance}")


def get_vectorizer_hash(model: str, prompt: str):
    hashed = hashlib.sha256((model + prompt).encode()).hexdigest()
    return "vectorizer:" + hashed


def get_prompt_hash(model: str, dataset_key: str, prompt: str, distance: str):
    hashed = hashlib.sha256(
        (model + dataset_key + prompt + distance).encode()
    ).hexdigest()
    return "prompt:" + hashed


def get_es_query_hash(index_name: str, dataset_key: str, query: str):
    hashed = hashlib.sha256((index_name + dataset_key + query).encode()).hexdigest()
    return "query:" + hashed


def get_reranker_hash(
    model: str, query: str, passage_ids: list, dataset_key: str, count: int
):
    hashed = hashlib.sha256(
        (model + query + str(passage_ids) + dataset_key + str(count)).encode()
    ).hexdigest()
    return "reranker:" + hashed


def get_relevant_document_count_hash(passage_id: str, dataset_key: str):
    hashed = hashlib.sha256((passage_id + dataset_key).encode()).hexdigest()
    return "count:" + hashed


def get_dataset_key(dataset_name: str, split: str):
    return replace_slash_with_dash(f"{dataset_name}-{split}")


def get_all_es_index_combinations():
    dataset_keys = [
        get_dataset_key(dataset_name, split)
        for dataset_name in DATASET_NAMES
        for split, _ in CHUNK_SIZES
    ]

    return [
        (index, dataset_key) for index in INDEX_NAMES for dataset_key in dataset_keys
    ]


def get_all_qdrant_model_combinations():
    dataset_keys = [
        get_dataset_key(dataset_name, split)
        for dataset_name in DATASET_NAMES
        for split, _ in CHUNK_SIZES
    ]

    return [
        (model, distance, dataset_key)
        for dataset_key in dataset_keys
        for model in MODEL_NAMES
        for distance in DISTANCES
    ]


def get_all_openai_model_combinations():
    dataset_keys = [
        get_dataset_key(dataset_name, split)
        for dataset_name in DATASET_NAMES
        for split, _ in CHUNK_SIZES
    ]

    return [
        (model, distance, dataset_key)
        for dataset_key in dataset_keys
        for model in OPENAI_EMBEDDING_MODEL_NAMES
        for distance in DISTANCES
    ]


def get_query_with_prefix(query: str, prefix: str):
    return f"{prefix}{query}"
