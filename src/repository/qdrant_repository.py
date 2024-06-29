from typing import List, Tuple
from cache.cache import Cache
from common.passage import Passage
from common.utils import get_prompt_hash
from repository.repository import Repository
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct
from vectorizer.vectorizer import Vectorizer
import uuid
import json


class QdrantRepository(Repository):

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        model_name: str,
        vectors_config: VectorParams,
        vectorizer: Vectorizer,
        cache: Cache,
    ):
        self.qdrant = client
        self.collection_name = collection_name
        self.model_name = model_name
        self.vectorizer = vectorizer
        self.cache = cache

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
        hash_key = get_prompt_hash(self.model_name, query)

        cached_value = self.cache.get(hash_key)

        if cached_value:
            dicts = json.loads(cached_value)
            passages = [Passage.from_dict(d) for d in dicts]
            return passages

        vector = self.vectorizer.get_vector(query)

        data = self.qdrant.search(
            collection_name=self.collection_name, query_vector=vector, limit=10
        )

        passages = list(
            map(
                lambda point: Passage(
                    point.payload["id"],
                    point.payload["text"],
                    point.payload["metadata"]["title"],
                    point.payload["metadata"]["start_index"],
                ),
                data,
            )
        )

        result_json = json.dumps([p.dict() for p in passages])
        self.cache.set(hash_key, result_json)

        return passages

    def delete(self, query):
        return self.qdrant.delete(query)
