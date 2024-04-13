from abc import ABC, abstractmethod
from common.document import Document

class Repository(ABC): 
    @abstractmethod
    def insertOne(self, data: Document):
        pass

    @abstractmethod
    def insertMany(self, data: list[Document]):
        pass

    @abstractmethod
    def find(self, query):
        pass

    @abstractmethod
    def delete(self, query):
        pass