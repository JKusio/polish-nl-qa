from repository.qdrant_repository import QdrantRepository
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from vectorizer.hf_vectorizer import HFVectorizer

def main():
    client = QdrantClient(host='localhost', port=6333)
    vectorizer = HFVectorizer("sdadas/mmlw-retrieval-roberta-large")
    qdrant_repository = QdrantRepository(client, "test", VectorParams(size=1024, distance=Distance.COSINE), vectorizer)

main()