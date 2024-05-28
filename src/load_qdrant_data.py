from langchain_text_splitters import RecursiveCharacterTextSplitter
from numpy import character, vectorize
from common.models_dimensions import MODEL_DIMENSIONS_MAP
from common.passage import Passage
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


# Models to use
# sdadas/mmlw-retrieval-roberta-large
# ipipan/silver-retriever-base-v1
# intfloat/multilingual-e5-large
# sdadas/mmlw-roberta-large
# BAAI/bge-m3 (dense)


def main():
    client = QdrantClient(host="localhost", port=6333)

    dataset_names = ["ipipan/polqa", "clarin-pl/poquad"]

    model_names = [
        "sdadas/mmlw-retrieval-roberta-large",
        "ipipan/silver-retriever-base-v1",
        "intfloat/multilingual-e5-large",
        "sdadas/mmlw-roberta-large",
        "BAAI/bge-m3",
    ]

    distances = [Distance.COSINE, Distance.EUCLID]

    for dataset_name in dataset_names:
        for model_name in model_names:
            vectorizer = HFVectorizer(model_name)
            embeddings = HuggingFaceEmbeddings(model_name=model_name)
            dimension = MODEL_DIMENSIONS_MAP[model_name]

            for distance in distances:
                get_interquartile_data(
                    client,
                    dataset_name,
                    model_name,
                    vectorizer,
                    embeddings,
                    distance,
                    dimension,
                )

                get_standard_deviation_data(
                    client,
                    dataset_name,
                    model_name,
                    vectorizer,
                    embeddings,
                    distance,
                    dimension,
                )

                get_percentile_data(
                    client,
                    dataset_name,
                    model_name,
                    vectorizer,
                    embeddings,
                    distance,
                    dimension,
                )

                get_character_splitting_data(
                    client,
                    dataset_name,
                    model_name,
                    vectorizer,
                    500,
                    100,
                    distance,
                    dimension,
                )

                get_character_splitting_data(
                    client,
                    dataset_name,
                    model_name,
                    vectorizer,
                    1000,
                    200,
                    distance,
                    dimension,
                )

                get_character_splitting_data(
                    client,
                    dataset_name,
                    model_name,
                    vectorizer,
                    2000,
                    400,
                    distance,
                    dimension,
                )


def get_interquartile_data(
    client: QdrantClient,
    dataset_name: str,
    model_name: str,
    vectorizer: HFVectorizer,
    embeddings: HuggingFaceEmbeddings,
    distance: Distance,
    size: int,
):
    collection_name = get_qdrant_collection_name(
        dataset_name, model_name, "interquartile", 1.5, distance
    )
    repository = QdrantRepository(
        client,
        collection_name,
        VectorParams(size=size, distance=distance),
        vectorizer,
    )
    text_splitter = SemanticChunker(
        embeddings, breakpoint_threshold_type="interquartile"
    )

    dataset_getter = (
        dataset_name == "ipipan/polqa" and PolqaDatasetGetter() or PoquadDatasetGetter()
    )
    passage_factory = PassageFactory(text_splitter, dataset_getter)

    data_importer = QdrantDataImporter(repository, passage_factory)
    data_importer.import_data()


def get_standard_deviation_data(
    client: QdrantClient,
    dataset_name: str,
    model_name: str,
    vectorizer: HFVectorizer,
    embeddings: HuggingFaceEmbeddings,
    distance: Distance,
    size: int,
):
    collection_name = get_qdrant_collection_name(
        dataset_name, model_name, "standard_deviation", 3, distance
    )
    repository = QdrantRepository(
        client,
        collection_name,
        VectorParams(size=size, distance=distance),
        vectorizer,
    )
    text_splitter = SemanticChunker(
        embeddings, breakpoint_threshold_type="standard_deviation"
    )

    dataset_getter = (
        dataset_name == "ipipan/polqa" and PolqaDatasetGetter() or PoquadDatasetGetter()
    )
    passage_factory = PassageFactory(text_splitter, dataset_getter)

    data_importer = QdrantDataImporter(repository, passage_factory)
    data_importer.import_data()


def get_percentile_data(
    client: QdrantClient,
    dataset_name: str,
    model_name: str,
    vectorizer: HFVectorizer,
    embeddings: HuggingFaceEmbeddings,
    distance: Distance,
    size: int,
):
    collection_name = get_qdrant_collection_name(
        dataset_name, model_name, "percentile", 95, distance
    )
    repository = QdrantRepository(
        client,
        collection_name,
        VectorParams(size=size, distance=distance),
        vectorizer,
    )
    text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")

    dataset_getter = (
        dataset_name == "ipipan/polqa" and PolqaDatasetGetter() or PoquadDatasetGetter()
    )
    passage_factory = PassageFactory(text_splitter, dataset_getter)

    data_importer = QdrantDataImporter(repository, passage_factory)
    data_importer.import_data()


def get_character_splitting_data(
    client: QdrantClient,
    dataset_name: str,
    model_name: str,
    vectorizer: HFVectorizer,
    chunk_size: int,
    chunk_overlap: int,
    distance: Distance,
    size: int,
):
    collection_name = get_qdrant_collection_name(
        dataset_name, model_name, "character_splitting", chunk_size, distance
    )
    repository = QdrantRepository(
        client,
        collection_name,
        VectorParams(size=size, distance=distance),
        vectorizer,
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, strip_whitespace=True
    )

    dataset_getter = (
        dataset_name == "ipipan/polqa" and PolqaDatasetGetter() or PoquadDatasetGetter()
    )
    passage_factory = PassageFactory(text_splitter, dataset_getter, chunk_overlap)

    data_importer = QdrantDataImporter(repository, passage_factory)
    data_importer.import_data()


main()
