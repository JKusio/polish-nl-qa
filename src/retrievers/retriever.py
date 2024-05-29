from abc import ABC, abstractmethod
from typing import List


class Retriever(ABC):
    @abstractmethod
    def get_relevant_passages(self, query: str) -> List[str]:
        pass
