from abc import ABC, abstractmethod
from common.result import Result


class Reranker(ABC):
    @abstractmethod
    def rerank(self, result: Result, count: int) -> Result:
        pass
