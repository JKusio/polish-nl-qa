from abc import ABC, abstractmethod
from typing import Any, List, Tuple
from common.passage import Passage


class Repository(ABC):
    @abstractmethod
    def insert_one(self, data: Passage):
        pass

    @abstractmethod
    def insert_one_with_vector(self, data: Passage, vector: Any):
        pass

    @abstractmethod
    def insert_many(self, data: List[Passage]):
        pass

    @abstractmethod
    def insert_many_with_vectors(self, data: List[Tuple]):
        pass

    @abstractmethod
    def find(self, query) -> List[Passage]:
        pass

    @abstractmethod
    def delete(self, query):
        pass
