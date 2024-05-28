from abc import ABC, abstractmethod
from common.passage import Passage


class Repository(ABC):
    @abstractmethod
    def insert_one(self, data: Passage):
        pass

    @abstractmethod
    def insert_many(self, data: list[Passage]):
        pass

    @abstractmethod
    def find(self, query) -> list[Passage]:
        pass

    @abstractmethod
    def delete(self, query):
        pass
