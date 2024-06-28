from typing import List
from elasticsearch import Elasticsearch
from cache.cache import Cache
from common.passage import Passage
from common.utils import get_es_query_hash
from repository.repository import Repository
import json


class ESRepository(Repository):
    def __init__(self, client: Elasticsearch, index_name: str, cache: Cache):
        self.client = client
        self.index_name = index_name
        self.cache = cache

    def insert_one(self, data: Passage):
        return self.client.index(index=self.index_name, body=self._map_data(data))

    def insert_many(self, data: Passage):
        body = [
            {"index": {"_index": self.index_name}, "body": self._map_data(doc)}
            for doc in data
        ]
        return self.client.bulk(body=body)

    def _map_data(self, data: Passage):
        return {
            "id": data.id,
            "text": data.text,
            "title": data.title,
            "start_index": data.start_index,
        }

    def find(self, query: str, dataset_key: str) -> List[Passage]:
        hash_key = get_es_query_hash(self.index_name, dataset_key, query)
        cached_value = self.cache.get(hash_key)

        if cached_value:
            dicts = json.loads(cached_value)
            passages = [Passage.from_dict(d) for d in dicts]
            return passages

        body = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"text": query}},
                        {"match": {"dataset_key": dataset_key}},
                    ]
                }
            }
        }

        result = self.client.search(index=self.index_name, body=body)
        passages = [
            Passage(
                hit["_source"]["id"],
                hit["_source"]["text"],
                hit["_source"]["title"],
                hit["_source"]["start_index"],
            )
            for hit in result["hits"]["hits"]
        ]

        result_json = json.dumps([p.dict() for p in passages])
        self.cache.set(hash_key, result_json)

        return passages

    def delete(self, query: str):
        body = {"query": {"match": {"text": query}}}
        return self.client.delete_by_query(index=self.index_name, body=body)
