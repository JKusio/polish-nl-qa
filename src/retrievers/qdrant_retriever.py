from typing import List
from common.result import Result
from repository.qdrant_repository import QdrantRepository
from retrievers.retriever import Retriever


class QdrantRetriever(Retriever):
    def __init__(self, repository: QdrantRepository, dataset_key: str):
        self.repository = repository
        self.dataset_key = dataset_key

    def get_relevant_passages(self, query: str) -> Result:
        result = self.repository.find(query, self.dataset_key)

        return result
