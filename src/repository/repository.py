from abc import ABC, abstractmethod
from common.passage import Passage

class Repository(ABC): 
    @abstractmethod
    def insertOne(self, data: Passage):
        pass

    @abstractmethod
    def insertMany(self, data: list[Passage]):
        pass

    @abstractmethod
    def find(self, query) -> list[Passage]: 
        pass

    @abstractmethod
    def delete(self, query):
        pass