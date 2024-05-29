from typing import List, Tuple
from common.passage import Passage
from repository.repository import Repository
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct
from vectorizer.vectorizer import Vectorizer
import uuid


class QdrantRepository(Repository):
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        vectors_config: VectorParams,
        vectorizer: Vectorizer,
    ):
        self.qdrant = client
        self.collection_name = collection_name
        self.vectorizer = vectorizer

        collections = self.qdrant.get_collections()
        if collection_name not in [
            collection.name for collection in collections.collections
        ]:
            print(f"Collection {collection_name} not found. Creating collection...")
            self.qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
            )

        print(f"Qdrant collection {collection_name} repository initialized")

    def insert_one(self, passage: Passage):
        return self.qdrant.upsert(
            collection_name=self.collection_name,
            wait=False,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=self.vectorizer.get_vector(passage.text),
                    payload=passage.dict(),
                )
            ],
        )

    def insert_one_with_vector(self, data: Passage, vector):
        return self.qdrant.upsert(
            collection_name=self.collection_name,
            wait=False,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=data.dict(),
                )
            ],
        )

    def insert_many(self, passages: List[Passage]):
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=self.vectorizer.get_vector(passage.text),
                payload=passage.dict(),
            )
            for passage in passages
        ]

        return self.qdrant.upsert(
            collection_name=self.collection_name, wait=True, points=points
        )

    def insert_many_with_vectors(self, passages: List[Tuple]):
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=passage.dict(),
            )
            for (passage, vector) in passages
        ]

        return self.qdrant.upsert(
            collection_name=self.collection_name, wait=True, points=points
        )

    def find(self, query: str) -> List[Passage]:
        vector = self.vectorizer.get_vector(query)

        data = self.qdrant.search(
            collection_name=self.collection_name, query_vector=vector
        )

        return list(
            map(
                lambda point: Passage(
                    point.payload["id"],
                    point.payload["text"],
                    point.payload["metadata"]["title"],
                ),
                data,
            )
        )

    def delete(self, query):
        return self.qdrant.delete(query)
