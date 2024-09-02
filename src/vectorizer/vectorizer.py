from abc import ABC, abstractmethod
from typing import Any


class Vectorizer(ABC):
    @abstractmethod
    def get_vector(self, text: str) -> Any:
        pass

    @abstractmethod
    def get_similarity(self, vector1: Any, vector2: Any) -> float:
        pass
