from abc import ABC, abstractmethod
from typing import List

from common.result import Result


class Retriever(ABC):
    @abstractmethod
    def get_relevant_passages(self, query: str) -> Result:
        pass
