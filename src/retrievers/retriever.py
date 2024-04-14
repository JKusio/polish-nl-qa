from abc import ABC, abstractmethod

class Retriever(ABC):
    @abstractmethod
    def get_relevant_docs(self, query: str):
        pass