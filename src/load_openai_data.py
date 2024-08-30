import json
import os
from pyexpat import model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from cache.cache import Cache
from common.models_dimensions import MODEL_DIMENSIONS_MAP
from common.names import CHUNK_SIZES, DATASET_NAMES, DISTANCES
from common.passage_factory import PassageFactory
from common.qdrant_data_importer import QdrantDataImporter
from common.qdrant_openai_data_importer import QdrantOpenAIDataImporter
from common.utils import (
    get_qdrant_collection_name,
    get_vectorizer_hash,
    replace_slash_with_dash,
)
from dataset import poquad_dataset_getter
from dataset.poquad_dataset_getter import PoquadDatasetGetter
from repository.qdrant_openai_repository import QdrantOpenAIRepository
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from vectorizer.hf_vectorizer import HFVectorizer
from dataset.polqa_dataset_getter import PolqaDatasetGetter


def main():
    client = QdrantClient(host="localhost", port=6333)
    cache = Cache()

    for distance in DISTANCES:
        insert_passage_data(client, distance, cache)

    insert_query_cache(cache)


def get_passage_factory(
    chunk_size: int, chunk_overlap: int, dataset_name: str
) -> PassageFactory:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, strip_whitespace=True
    )

    dataset_getter = (
        dataset_name == "ipipan/polqa" and PolqaDatasetGetter() or PoquadDatasetGetter()
    )
    return PassageFactory(text_splitter, dataset_getter)


def insert_passage_data(
    client: QdrantClient,
    distance: Distance,
    cache: Cache,
):
    model_name = "text-embedding-3-large"

    passage_filenames = [
        f
        for f in os.listdir("openai_batches")
        if os.path.isfile(os.path.join("openai_batches", f))
    ]

    for dataset_name in DATASET_NAMES:
        for chunk_size, chunk_overlap in CHUNK_SIZES:
            passage_factory = get_passage_factory(
                chunk_size, chunk_overlap, dataset_name
            )

            repository = QdrantOpenAIRepository.get_repository(
                client,
                model_name,
                distance,
                cache,
            )

            data_importer = QdrantOpenAIDataImporter(repository, passage_factory)

            batch_filename = next(
                (
                    f
                    for f in passage_filenames
                    if (
                        model_name in f
                        and f"{chunk_size}_{chunk_overlap}" in f
                        and replace_slash_with_dash(dataset_name) in f
                    )
                ),
                None,
            )

            print(batch_filename)

            batch_data = []
            with open(
                f"openai_batches/{batch_filename}",
                "r",
                encoding="utf-8",
            ) as f:
                for line in f:
                    batch_data.append(json.loads(line))

            data_importer.import_data(batch_data)


def insert_query_cache(cache: Cache):
    model_name = "text-embedding-3-large"

    poquad_file = "openai_batches/clarin-pl-poquad.jsonl"
    polqa_file = "openai_batches/ipipan-polqa.jsonl"

    poquad_batch_data = []
    with open(poquad_file, "r", encoding="utf-8") as f:
        for line in f:
            poquad_batch_data.append(json.loads(line))

    polqa_batch_data = []
    with open(polqa_file, "r", encoding="utf-8") as f:
        for line in f:
            polqa_batch_data.append(json.loads(line))

    poquad_dataset_getter = PoquadDatasetGetter()
    polqa_dataset_getter = PolqaDatasetGetter()

    poquad_dataset = poquad_dataset_getter.get_test_dataset()
    polqa_dataset = polqa_dataset_getter.get_test_dataset()

    i = 0

    for entry in poquad_dataset:
        query = entry.question
        vector = next(
            (p for p in poquad_batch_data if p["custom_id"] == entry.id), None
        )["response"]["body"]["data"][0]["embedding"]

        cache.set(
            get_vectorizer_hash(model_name, query),
            json.dumps(vector),
        )
        i += 1
        print(f"Processed {i} queries")

    for entry in polqa_dataset:
        query = entry.question
        vector = next(
            (p for p in polqa_batch_data if int(p["custom_id"]) == entry.id), None
        )["response"]["body"]["data"][0]["embedding"]

        cache.set(
            get_vectorizer_hash(model_name, query),
            json.dumps(vector),
        )
        i += 1
        print(f"Processed {i} queries")


main()
