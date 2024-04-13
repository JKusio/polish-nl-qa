from repository.repository import Repository
from common.document import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

class QdrantRepository(Repository):
    def __init__(self, client: QdrantClient, collection_name: str):
        self.qdrant = client
        self.collection_name = collection_name

        collections = self.qdrant.get_collections()
        if collection_name not in [collection.name for collection in collections.collections]:
            print(f"Collection {collection_name} not found. Creating collection...")
            self.qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
        
        print(f"Qdrant collection {collection_name} repository initialized")

    def insertOne(self, data: Document):
        return self.qdrant.add(data)
    
    def insertMany(self, data: list[Document]):
        return super().insertMany(data)

    def find(self, query):
        return self.qdrant.search(query)

    def delete(self, query):
        return self.qdrant.delete(query)