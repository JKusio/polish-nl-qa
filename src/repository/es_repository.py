from typing import List
from elasticsearch import Elasticsearch, helpers
from cache.cache import Cache
from common.passage import Passage
from common.result import Result
from common.utils import (
    get_es_query_hash,
    get_relevant_document_count_hash,
)
from repository.repository import Repository
import json


class ESRepository(Repository):
    def __init__(self, client: Elasticsearch, index_name: str, cache: Cache):
        self.client = client
        self.index_name = index_name
        self.cache = cache

    def insert_one(self, data: Passage):
        return self.client.index(index=self.index_name, body=data.dict())

    def insert_many(self, data: list[Passage]):
        documents = [d.dict() for d in data]
        return helpers.bulk(self.client, documents, index=self.index_name)

    def find(self, query: str, dataset_key: str, size: int = 10) -> Result:
        hash_key = get_es_query_hash(self.index_name, dataset_key, query, size)
        cached_value = self.cache.get(hash_key)

        if cached_value:
            dicts = json.loads(cached_value)
            passages = [(Passage.from_dict(d["passage"]), d["score"]) for d in dicts]
            return Result(query, passages)

        body = {
            "size": size,
            "query": {
                "bool": {
                    "must": [
                        {"match": {"context": query}},
                        {"match": {"dataset_key": dataset_key}},
                    ]
                }
            },
        }

        result = self.client.search(index=self.index_name, body=body)

        if (len(result["hits"]["hits"])) == 0:
            return Result(query, [])

        max_score = result["hits"]["hits"][0]["_score"]
        min_score = result["hits"]["hits"][-1]["_score"]
        score_diff = max_score - min_score

        passages = [
            (
                Passage(
                    hit["_source"]["id"],
                    hit["_source"]["title"],
                    hit["_source"]["context"],
                    hit["_source"]["start_index"],
                    hit["_source"]["dataset"],
                    hit["_source"]["dataset_key"],
                    hit["_source"]["metadata"],
                ),
                1 if score_diff == 0 else (hit["_score"] - min_score) / score_diff,
            )
            for hit in result["hits"]["hits"]
        ]

        result_json = json.dumps(
            [{"passage": p.dict(), "score": s} for (p, s) in passages]
        )
        self.cache.set(hash_key, result_json)

        return Result(query, passages)

    def delete(self, query: str):
        body = {"query": {"match": {"text": query}}}
        return self.client.delete_by_query(index=self.index_name, body=body)

    def count_relevant_documents(self, passage_ids: list[str], dataset_key: str) -> int:
        sorted_passage_ids = sorted(passage_ids)
        joined_passage_ids = ",".join(map(str, sorted_passage_ids))
        hash_key = get_relevant_document_count_hash(joined_passage_ids, dataset_key)

        is_poquad = True if "poquad" in dataset_key else False

        must = [
            {"match": {"dataset_key": dataset_key}},
        ]

        if is_poquad:
            must.append({"match": {"id": passage_ids[0]}})
        else:
            must.append({"terms": {"metadata.passage_id": passage_ids}})

        body = {
            "query": {"bool": {"must": must}},
        }

        response = self.client.count(index=self.index_name, body=body)

        self.cache.set(hash_key, response["count"])

        return response["count"]
