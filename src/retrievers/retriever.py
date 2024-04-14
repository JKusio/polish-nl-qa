from abc import ABC, abstractmethod

class Retriever(ABC):
    @abstractmethod
    def get_relevant_passages(self, query: str) -> list[str]:
        pass