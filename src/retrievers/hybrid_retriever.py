from collections import defaultdict
from typing import List
from common.result import Result
from repository.es_repository import ESRepository
from repository.qdrant_repository import QdrantRepository
from retrievers.retriever import Retriever


class HybridRetriever(Retriever):
    def __init__(
        self,
        es_repository: ESRepository,
        qdrant_repository: QdrantRepository,
        dataset_key: str,
        alpha: float = 0.5,  # weight for ES
    ):
        self.es_repository = es_repository
        self.qdrant_repository = qdrant_repository
        self.dataset_key = dataset_key
        self.alpha = alpha

    def get_relevant_passages(self, query: str) -> List[str]:
        es_result = self.es_repository.find(query, self.dataset_key)
        qdrant_result = self.qdrant_repository.find(query, self.dataset_key)

        weighted_es_passages = [
            (passage, score * self.alpha) for passage, score in es_result.passages
        ]
        weighted_qdrant_passages = [
            (passage, score * (1 - self.alpha))
            for passage, score in qdrant_result.passages
        ]

        combined_scores = defaultdict(float)

        for passage, score in weighted_es_passages + weighted_qdrant_passages:
            combined_scores[passage] += score

        final_results = list(combined_scores.items())
        final_results.sort(key=lambda x: x[1], reverse=True)

        return Result(query, final_results[:10])
