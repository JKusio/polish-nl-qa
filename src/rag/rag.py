from abc import ABC, abstractmethod

from retrievers.retriever import Retriever
from vectorizer.vectorizer import Vectorizer

class RAG(ABC):
    @abstractmethod
    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    @abstractmethod
    def generate(self, query: str):
        pass