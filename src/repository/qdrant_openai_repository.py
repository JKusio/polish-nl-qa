from typing import List, Tuple
from cache.cache import Cache
from common.models_dimensions import MODEL_DIMENSIONS_MAP
from common.passage import Passage
from common.result import Result
from common.utils import (
    get_prompt_hash,
    get_qdrant_collection_name,
    get_relevant_document_count_hash,
)
from repository.repository import Repository
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, PointStruct, Distance
import uuid
import json

from vectorizer.openai_vectorizer import OpenAIVectorizer


class QdrantOpenAIRepository(Repository):

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        model_name: str,
        vectors_config: VectorParams,
        vectorizer: OpenAIVectorizer,
        cache: Cache,
    ):
        self.qdrant = client
        self.collection_name = collection_name
        self.model_name = model_name
        self.cache = cache
        self.distance = (
            Distance.COSINE
            if Distance.COSINE.lower() in collection_name.lower()
            else Distance.EUCLID
        )
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

        print(f"Qdrant openai collection {collection_name} repository initialized")

    def insert_one(self, passage: Passage):
        return self.qdrant.upsert(
            collection_name=self.collection_name,
            wait=False,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=self.vectorizer.get_vector(passage.context),
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
                vector=self.vectorizer.get_vector(passage.context),
                payload=passage.dict(),
            )
            for passage in passages
        ]

        return self.qdrant.upsert(
            collection_name=self.collection_name, wait=True, points=points
        )

    def insert_many_with_vectors(self, passages: List[Tuple[Passage, List[float]]]):
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

    def find(self, query: str, dataset_key: str, size: int = 10) -> Result:
        full_query = query
        hash_key = get_prompt_hash(
            self.model_name, dataset_key, full_query, self.distance, size
        )

        cached_value = self.cache.get(hash_key)

        if cached_value:
            dicts = json.loads(cached_value)
            passages = [(Passage.from_dict(d["passage"]), d["score"]) for d in dicts]
            return Result(query, passages)

        vector = self.vectorizer.get_vector(full_query)

        data = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=size,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="dataset_key",
                        match=models.MatchValue(value=dataset_key),
                    )
                ]
            ),
        )

        if (len(data)) == 0:
            return Result(query, [])

        max_score = data[0].score
        min_score = data[-1].score
        score_diff = max_score - min_score

        passages = [
            (
                Passage(
                    point.payload["id"],
                    point.payload["title"],
                    point.payload["context"],
                    point.payload["start_index"],
                    point.payload["dataset"],
                    point.payload["dataset_key"],
                    point.payload["metadata"],
                ),
                1 if score_diff == 0 else (point.score - min_score) / score_diff,
            )
            for point in data
        ]

        result_json = json.dumps(
            [{"passage": p.dict(), "score": s} for (p, s) in passages]
        )
        self.cache.set(hash_key, result_json)

        return Result(query, passages)

    def delete(self, query):
        return self.qdrant.delete(query)

    def get_repository(
        client: QdrantClient,
        model_name: str,
        distance: Distance,
        cache: Cache,
    ):
        collection_name = get_qdrant_collection_name(model_name, distance)
        vectorizer = OpenAIVectorizer(model_name, cache)

        return QdrantOpenAIRepository(
            client,
            collection_name,
            model_name,
            VectorParams(size=MODEL_DIMENSIONS_MAP[model_name], distance=distance),
            vectorizer,
            cache,
        )

    def count_relevant_documents(self, passage_ids, dataset_key) -> int:
        sorted_passage_ids = sorted(passage_ids)
        joined_passage_ids = ",".join(map(str, sorted_passage_ids))
        hash_key = get_relevant_document_count_hash(joined_passage_ids, dataset_key)

        cached_value = self.cache.get(hash_key)

        if cached_value:
            return int(cached_value)

        is_poquad = True if "poquad" in dataset_key else False

        must = [
            models.FieldCondition(
                key="dataset_key",
                match=models.MatchValue(value=dataset_key),
            )
        ]

        if is_poquad:
            must.append(
                models.FieldCondition(
                    key="id",
                    match=models.MatchAny(any=sorted_passage_ids),
                )
            )
        else:
            must.append(
                models.FieldCondition(
                    key="metadata.passage_id",
                    match=models.MatchAny(any=sorted_passage_ids),
                )
            )

        result = self.qdrant.count(
            collection_name=self.collection_name,
            count_filter=models.Filter(must=must),
            exact=True,
        )

        self.cache.set(hash_key, str(result.count))

        return result.count
