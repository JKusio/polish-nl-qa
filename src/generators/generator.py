from abc import ABC, abstractmethod
from common.passage import Passage


class Generator(ABC):
    @abstractmethod
    def generate_answer(self, query: str, passages: list[Passage]) -> str:
        pass
