from typing import List
from common.result import Result
from repository.qdrant_repository import QdrantRepository
from rerankers.hf_reranker import HFReranker
from retrievers.retriever import Retriever


class QdrantRetriever(Retriever):
    def __init__(
        self,
        repository: QdrantRepository,
        dataset_key: str,
        reranker: HFReranker = None,
    ):
        self.repository = repository
        self.dataset_key = dataset_key
        self.reranker = reranker

    def get_relevant_passages(self, query: str, size: int = 10) -> Result:
        docs_size = size * 2 if self.reranker else size

        result = self.repository.find(query, self.dataset_key, docs_size)

        if self.reranker:
            result = self.reranker.rerank(result, size, self.dataset_key)

        return result
