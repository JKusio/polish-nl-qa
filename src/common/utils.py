import string
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


def get_prompt_hash(
    model: str, dataset_key: str, prompt: str, distance: str, size: int
):
    hashed = hashlib.sha256(
        (model + dataset_key + prompt + distance + str(size)).encode()
    ).hexdigest()
    return "prompt:" + hashed


def get_es_query_hash(index_name: str, dataset_key: str, query: str, size: int):
    hashed = hashlib.sha256(
        (index_name + dataset_key + query + str(size)).encode()
    ).hexdigest()
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


def get_generator_hash(query: str, context: str, type: str, model: str):
    hashed = hashlib.sha256((query + context + type + model).encode()).hexdigest()
    return "generator:" + hashed


def get_ner_hash(answer: str, context: str):
    hashed = hashlib.sha256((answer + context).encode()).hexdigest()
    return "ner:" + hashed


def get_halucination_hash(answer: str, context: str):
    hashed = hashlib.sha256((answer + context).encode()).hexdigest()
    return "halucination:" + hashed


def get_answer_reranker_hash(answer: str, passages: list[Passage]):
    hashed = hashlib.sha256((answer + str(passages)).encode()).hexdigest()
    return "answer_reranker:" + hashed


def get_query_reranker_hash(query: str, answer: str):
    hashed = hashlib.sha256((query + answer).encode()).hexdigest()
    return "query_reranker:" + hashed


def get_query_to_passages_reranker_hash(query: str, passages: list[Passage]):
    hashed = hashlib.sha256((query + str(passages)).encode()).hexdigest()
    return "query_to_passages_reranker:" + hashed


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


def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def split_text_into_token_chunks(text, tokenizer, max_length, overlap):
    tokens = tokenizer.encode(text, add_special_tokens=False)

    chunks = []
    for i in range(0, len(tokens), max_length - overlap):
        chunk = tokens[i : i + max_length]
        chunks.append(chunk)

        if len(chunk) < max_length:
            break

    text_chunks = [tokenizer.decode(chunk) for chunk in chunks]

    return text_chunks


def split_into_chunks(text, max_length, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_length
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap  # overlap for context
    return chunks
