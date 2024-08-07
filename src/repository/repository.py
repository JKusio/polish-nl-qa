from abc import ABC, abstractmethod
from typing import Any, List, Tuple
from common.passage import Passage
from common.result import Result


class Repository(ABC):
    @abstractmethod
    def insert_one(self, data: Passage):
        pass

    @abstractmethod
    def insert_many(self, data: List[Passage]):
        pass

    @abstractmethod
    def find(self, query, dataset_key) -> Result:
        pass

    @abstractmethod
    def delete(self, query):
        pass

    @abstractmethod
    def count_relevant_documents(self, title, dataset_key) -> int:
        pass
