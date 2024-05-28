import uuid
from common.passage import Passage


def get_passages_for_embedding(dataset):
    unique_contexts = set((row["title"], row["context"]) for row in dataset)

    return list(
        map(lambda row: Passage(generate_id(), row[1], row[0]), unique_contexts)
    )


def generate_id():
    return str(uuid.uuid4())


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
