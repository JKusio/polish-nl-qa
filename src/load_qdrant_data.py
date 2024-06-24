from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from common.models_dimensions import MODEL_DIMENSIONS_MAP
from common.names import (
    CHUNK_SIZES,
    DATASET_NAMES,
    DISTANCES,
    MODEL_NAMES,
    SEMANTIC_TYPES,
)
from common.passage_factory import PassageFactory
from common.qdrant_data_importer import QdrantDataImporter
from common.utils import get_qdrant_collection_name
from dataset.poquad_dataset_getter import PoquadDatasetGetter
from repository.qdrant_repository import QdrantRepository
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from vectorizer.hf_vectorizer import HFVectorizer
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from dataset.polqa_dataset_getter import PolqaDatasetGetter


def main():
    client = QdrantClient(host="localhost", port=6333)

    for dataset_name in DATASET_NAMES:
        for model_name in MODEL_NAMES:
            vectorizer = HFVectorizer(model_name)
            embeddings = HuggingFaceEmbeddings(model_name=model_name)
            dimension = MODEL_DIMENSIONS_MAP[model_name]
            for chunk_size, chunk_overlap in CHUNK_SIZES:
                insert_character_splitting_passage_data(
                    client,
                    dataset_name,
                    model_name,
                    vectorizer,
                    DISTANCES,
                    chunk_size,
                    chunk_overlap,
                    dimension,
                )

            for semantic_type in SEMANTIC_TYPES:
                insert_semantic_passage_data(
                    client,
                    dataset_name,
                    model_name,
                    embeddings,
                    vectorizer,
                    DISTANCES,
                    semantic_type,
                    dimension,
                )


def insert_semantic_passage_data(
    client: QdrantClient,
    dataset_name: str,
    model_name: str,
    embeddings: HuggingFaceEmbeddings,
    vectorizer: HFVectorizer,
    distances: List[Distance],
    breakpoint_threshold_type: str,
    dimension: int,
):
    repositories = []

    for distance in distances:
        collection_name = get_qdrant_collection_name(
            dataset_name, model_name, breakpoint_threshold_type, 1.5, distance
        )

        repository = QdrantRepository(
            client,
            collection_name,
            VectorParams(size=dimension, distance=distance),
            vectorizer,
        )

        repositories.append(repository)

    passage_factory = get_semantic_passage_factory(
        embeddings, breakpoint_threshold_type, dataset_name
    )

    data_importer = QdrantDataImporter(repositories, passage_factory, vectorizer)

    data_importer.import_data()


def get_semantic_passage_factory(
    embeddings: HuggingFaceEmbeddings,
    breakpoint_threshold_type: str,
    dataset_name: str,
) -> PassageFactory:
    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
    )

    dataset_getter = (
        dataset_name == "ipipan/polqa" and PolqaDatasetGetter() or PoquadDatasetGetter()
    )
    return PassageFactory(text_splitter, dataset_getter)


def insert_character_splitting_passage_data(
    client: QdrantClient,
    dataset_name: str,
    model_name: str,
    vectorizer: HFVectorizer,
    distances: List[Distance],
    chunk_size: int,
    chunk_overlap: int,
    dimension: int,
):
    repositories = []

    for distance in distances:
        collection_name = get_qdrant_collection_name(
            dataset_name, model_name, "character", chunk_size, distance
        )

        repository = QdrantRepository(
            client,
            collection_name,
            VectorParams(size=dimension, distance=distance),
            vectorizer,
        )

        repositories.append(repository)

    passage_factory = get_character_splitting_passage_factory(
        chunk_size, chunk_overlap, dataset_name
    )

    data_importer = QdrantDataImporter(repositories, passage_factory, vectorizer)

    data_importer.import_data()


def get_character_splitting_passage_factory(
    chunk_size: int, chunk_overlap: int, dataset_name: str
) -> PassageFactory:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, strip_whitespace=True
    )

    dataset_getter = (
        dataset_name == "ipipan/polqa" and PolqaDatasetGetter() or PoquadDatasetGetter()
    )
    return PassageFactory(text_splitter, dataset_getter, chunk_overlap)


main()
