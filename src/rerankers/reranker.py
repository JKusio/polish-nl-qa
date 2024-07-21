from abc import ABC, abstractmethod
from typing import List
from common.passage import Passage


class Reranker(ABC):
    @abstractmethod
    def get_relevant_passages(
        self, query: str, passages: List[Passage], count: int
    ) -> List[Passage]:
        pass
