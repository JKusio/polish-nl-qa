from common.utils import get_passages_for_embedding
from rag.hd_qdrant_rag import HDQdrantRAG
from repository.qdrant_repository import QdrantRepository
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from retrievers.qdrant_retriever import QdrantRetriever
from vectorizer.hf_vectorizer import HFVectorizer
from datasets import load_dataset

def main():
    client = QdrantClient(host='localhost', port=6333)
    vectorizer = HFVectorizer("sdadas/mmlw-retrieval-roberta-large")
    qdrant_repository = QdrantRepository(client, "test", VectorParams(size=1024, distance=Distance.COSINE), vectorizer)
    qdrant_retriever = QdrantRetriever(qdrant_repository)

    hd_qdrant_rag = HDQdrantRAG(qdrant_retriever, './output/roberta-base-squad2-pl/checkpoint-8500')
    result = hd_qdrant_rag.generate("Kto wymyślił ALF?")
    print(result)

main()