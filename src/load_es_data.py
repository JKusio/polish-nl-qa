from elasticsearch import Elasticsearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from cache.cache import Cache
from common.es_data_importer import ESDataImporter
from common.names import (
    CHUNK_SIZES,
    DATASET_NAMES,
    INDEX_NAMES,
)
from common.passage_factory import PassageFactory
from dataset.poquad_dataset_getter import PoquadDatasetGetter
from repository.es_repository import ESRepository
from dataset.polqa_dataset_getter import PolqaDatasetGetter


def main():
    client = Elasticsearch(
        hosts=["http://localhost:9200"],
    )
    cache = Cache()

    for index_name in INDEX_NAMES:
        insert_passage_data(client, index_name, cache)


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
    client: Elasticsearch,
    index_name: str,
    cache: Cache,
):
    for dataset_name in DATASET_NAMES:
        for chunk_size, chunk_overlap in CHUNK_SIZES:
            passage_factory = get_passage_factory(
                chunk_size, chunk_overlap, dataset_name
            )

            repository = ESRepository(client, index_name, cache)

            data_importer = ESDataImporter(repository, passage_factory)

            data_importer.import_data()


main()
