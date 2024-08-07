from typing import List
from common.result import Result
from repository.es_repository import ESRepository
from rerankers.hf_reranker import HFReranker
from retrievers.retriever import Retriever


class ESRetriever(Retriever):
    def __init__(
        self, repository: ESRepository, dataset_key: str, reranker: HFReranker = None
    ):
        self.repository = repository
        self.dataset_key = dataset_key
        self.reranker = reranker

    def get_relevant_passages(self, query: str) -> Result:
        result = self.repository.find(query, self.dataset_key)

        if self.reranker:
            result = self.reranker.rerank(result, 10, self.dataset_key)

        return result
