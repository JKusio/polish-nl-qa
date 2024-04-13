from repository.qdrant_repository import QdrantRepository
from qdrant_client import QdrantClient

def main():
    client = QdrantClient(host='localhost', port=6333)
    qdrant_repository = QdrantRepository(client, "test")



main()