from typing import List
from elasticsearch import Elasticsearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from common.models_dimensions import MODEL_DIMENSIONS_MAP
from common.names import (
    CHARACTER_SPLITTING_FUNCTION,
    DATASET_NAMES,
    MODEL_NAMES,
    SEMANTIC_TYPES,
    INDEX_NAMES,
)
from common.passage_factory import PassageFactory
from common.qdrant_data_importer import QdrantDataImporter
from common.utils import get_qdrant_collection_name, replace_slash_with_dash
from dataset.poquad_dataset_getter import PoquadDatasetGetter
from repository.qdrant_repository import QdrantRepository
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from vectorizer.hf_vectorizer import HFVectorizer
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from dataset.polqa_dataset_getter import PolqaDatasetGetter


def main():
    es_client = Elasticsearch(
        hosts=["http://localhost:9200"],
    )

    qdrant_client = QdrantClient(host="localhost", port=6333)

    qdrant_collections = qdrant_client.get_collections()

    for dataset_name in DATASET_NAMES:
        for split in CHARACTER_SPLITTING_FUNCTION:
            collection = next(
                (
                    c
                    for c in qdrant_collections.collections
                    if replace_slash_with_dash(dataset_name) in c.name
                    and replace_slash_with_dash(split) in c.name
                ),
                None,
            )

            if not collection:
                continue

            all_data, _ = qdrant_client.scroll(
                collection_name=collection.name, limit=10000
            )

            for index in INDEX_NAMES:
                insert_passages(
                    es_client,
                    index,
                    replace_slash_with_dash(f"{dataset_name}-{split}"),
                    all_data,
                )
                print("Inserted data for", dataset_name, split, index)

        for model_name in MODEL_NAMES:
            for split in SEMANTIC_TYPES:
                collection = next(
                    (
                        c
                        for c in qdrant_collections.collections
                        if (
                            replace_slash_with_dash(dataset_name) in c.name
                            and replace_slash_with_dash(model_name) in c.name
                            and replace_slash_with_dash(split) in c.name
                        )
                    ),
                    None,
                )

                if not collection:
                    continue

                all_data, _ = qdrant_client.scroll(
                    collection_name=collection.name, limit=10000
                )

                for index in INDEX_NAMES:
                    insert_passages(
                        es_client,
                        index,
                        replace_slash_with_dash(f"{dataset_name}-{model_name}-{split}"),
                        all_data,
                    )
                    print("Inserted data for", dataset_name, model_name, split, index)

    # I need for each index
    # for each dataset
    # 500
    # 1000
    # 2000
    # semantic for each model of the dataset
    # models
    # "sdadas/mmlw-retrieval-roberta-large",
    # "ipipan/silver-retriever-base-v1",
    # "intfloat/multilingual-e5-large",
    # "sdadas/mmlw-roberta-large",
    # "BAAI/bge-m3",
    # so it's 16 per index


def insert_passages(
    es_client: Elasticsearch,
    index_name: str,
    dataset_key: str,
    data: List[dict],
):
    for point in data:
        es_client.index(
            index=index_name,
            body={
                "dataset_key": dataset_key,
                "text": point.payload.get("text"),
                "title": point.payload.get("metadata").get("title"),
                "id": point.payload.get("id"),
                "start_index": point.payload.get("metadata").get("start_index"),
            },
        )


main()
