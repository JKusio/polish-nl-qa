from ast import Dict
from tkinter import Place
from xml.etree.ElementInclude import include
from elasticsearch import Elasticsearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from common.names import DATASET_NAMES, INDEX_NAMES
from common.passage import Passage
from common.passage_factory import PassageFactory
from common.utils import replace_slash_with_dash
from dataset.poquad_dataset_getter import PoquadDatasetGetter
from repository.es_repository import ESRepository
from repository.qdrant_repository import QdrantRepository
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from vectorizer.hf_vectorizer import HFVectorizer
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from dataset.polqa_dataset_getter import PolqaDatasetGetter
from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient
from cache.cache import Cache
from common.models_dimensions import MODEL_DIMENSIONS_MAP
from common.names import DISTANCES, MODEL_NAMES
from common.utils import get_all_qdrant_collection_names
from repository.qdrant_repository import QdrantRepository
from vectorizer.hf_vectorizer import HFVectorizer


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


def main():
    factory = get_passage_factory(1000, 100, DATASET_NAMES[1])

    passages = factory.get_passages()
    print(passages[1])


main()
