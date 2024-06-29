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
    def find(self, query) -> Result:
        pass

    @abstractmethod
    def delete(self, query):
        pass
