from typing import List, Tuple
from common.passage import Passage


class Result:
    def __init__(self, query: str, passages: List[Tuple[Passage, int]]) -> None:
        self.query = query
        self.passages = passages
