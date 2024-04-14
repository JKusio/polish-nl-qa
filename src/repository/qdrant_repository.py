from common.passage import Passage
from repository.repository import Repository
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct
from vectorizer.vectorizer import Vectorizer

class QdrantRepository(Repository):
    def __init__(self, client: QdrantClient, collection_name: str, vectors_config: VectorParams, vectorizer: Vectorizer):
        self.qdrant = client
        self.collection_name = collection_name
        self.vectorizer = vectorizer

        collections = self.qdrant.get_collections()
        if collection_name not in [collection.name for collection in collections.collections]:
            print(f"Collection {collection_name} not found. Creating collection...")
            self.qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
            )
        
        print(f"Qdrant collection {collection_name} repository initialized")

    def insertOne(self, passage: Passage):
        return self.qdrant.upsert(
            collection_name=self.collection_name,
            wait=False,
            points=[
                PointStruct(id=passage.id, vector=self.vectorizer.get_vector(passage.text), payload=passage.dict())
            ]
        )
    
    def insertMany(self, passages: list[Passage]):
        points = list(map(lambda passage: PointStruct(id=passage.id, vector=self.vectorizer.get_vector(passage.text), payload=passage.dict()), passages))

        return self.qdrant.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=points
        )

    def find(self, query: str):
        vector = self.vectorizer.get_vector(query)

        return self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=vector
        )

    def delete(self, query):
        return self.qdrant.delete(query)