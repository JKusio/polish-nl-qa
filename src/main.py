from tkinter import Place
from xml.etree.ElementInclude import include
from elasticsearch import Elasticsearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from common.passage import Passage
from common.passage_factory import PassageFactory
from dataset.poquad_dataset_getter import PoquadDatasetGetter
from repository.es_repository import ESRepository
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
    # es_client = Elasticsearch(
    #     hosts=["http://localhost:9200"],
    # )

    # es_repo = ESRepository(es_client, "basic_index")
    # passage = Passage("1", "context", "title", 0)

    # es_repo.insert_one(passage)

    # print(es_repo.find("context"))
    client = QdrantClient(host="localhost", port=6333)

    # get all collections
    collections = client.get_collections()

    poquad_ids = []

    print(collections.collections[0])

    for collection in collections.collections:
        if "poquad" in collection.name:
            print(collection.name)

            all_points = []
            next_page_offset = 0
            count = client.count(collection_name=collection.name)
            print(count.count)

            while True:
                # Scroll through the collection
                points, _ = client.scroll(
                    collection_name=collection.name,
                    offset=next_page_offset,
                )

                # Append fetched points to the list
                all_points.extend(points)

                next_page_offset += 10
                if next_page_offset > count.count - 9:
                    break

            # Extract ids from the points
            poquad_ids.extend([point.payload.get("text") for point in all_points])

    print(len(poquad_ids))

    poquad_set = set(poquad_ids)
    print(len(poquad_set))


main()
