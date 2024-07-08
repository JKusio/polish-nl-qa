from langchain_text_splitters import RecursiveCharacterTextSplitter
from cache.cache import Cache
from common.models_dimensions import MODEL_DIMENSIONS_MAP
from common.names import (
    CHUNK_SIZES,
    DATASET_NAMES,
    DISTANCES,
    MODEL_NAMES,
)
from common.passage_factory import PassageFactory
from common.qdrant_data_importer import QdrantDataImporter
from common.utils import get_qdrant_collection_name
from dataset.poquad_dataset_getter import PoquadDatasetGetter
from repository.qdrant_repository import QdrantRepository
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from vectorizer.hf_vectorizer import HFVectorizer
from dataset.polqa_dataset_getter import PolqaDatasetGetter


def main():
    client = QdrantClient(host="localhost", port=6333)
    cache = Cache()

    for model_name in MODEL_NAMES:
        vectorizer = HFVectorizer(model_name, cache)
        for distance in DISTANCES:
            insert_passage_data(client, model_name, distance, cache, vectorizer)


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
    model_name: str,
    distance: Distance,
    cache: Cache,
    vectorizer: HFVectorizer,
):
    collection_name = get_qdrant_collection_name(model_name, distance)

    for dataset_name in DATASET_NAMES:
        for chunk_size, chunk_overlap in CHUNK_SIZES:
            passage_factory = get_passage_factory(
                chunk_size, chunk_overlap, dataset_name
            )

            repository = QdrantRepository(
                client,
                collection_name,
                model_name,
                VectorParams(size=MODEL_DIMENSIONS_MAP[model_name], distance=distance),
                vectorizer,
                cache,
            )

            data_importer = QdrantDataImporter(repository, passage_factory, vectorizer)

            data_importer.import_data()


main()
