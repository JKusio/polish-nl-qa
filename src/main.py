from ast import Dict
from tkinter import Place
from xml.etree.ElementInclude import include
from elasticsearch import Elasticsearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from common.names import INDEX_NAMES
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


# Models to use
# sdadas/mmlw-retrieval-roberta-large
# ipipan/silver-retriever-base-v1
# intfloat/multilingual-e5-large
# sdadas/mmlw-roberta-large
# BAAI/bge-m3 (dense)


def main():
    es_client = Elasticsearch(
        hosts=["http://localhost:9200"],
    )
    cache = Cache()
    es_repositories: Dict[str, ESRepository] = {}

    for index_name in INDEX_NAMES:
        es_repositories[index_name] = ESRepository(es_client, index_name, cache)
        
    result = es_repositories["polish_index"].find(
    "Czym są pisma rabiniczne?", replace_slash_with_dash(f"{"clarin-pl-poquad"}-{"character-500"}")
    )
    
    print(result[0])


main()
