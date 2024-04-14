from abc import ABC, abstractmethod

class Vectorizer(ABC):
    @abstractmethod
    def get_vector(self, text: str):
        pass