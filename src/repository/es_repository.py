from elasticsearch import Elasticsearch
from common.passage import Passage
from repository.repository import Repository


class ESRepository(Repository):
    def __init__(self, client: Elasticsearch, index_name: str):
        self.client = client
        self.index_name = index_name

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

    def find(self, query: str):
        body = {"query": {"match": {"text": query}}}
        return self.client.search(index=self.index_name, body=body)

    def delete(self, query: str):
        body = {"query": {"match": {"text": query}}}
        return self.client.delete_by_query(index=self.index_name, body=body)
