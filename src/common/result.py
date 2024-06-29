from typing import List
from common.passage import Passage


class Result:
    def __init__(self, query: str, passages: List[Passage]) -> None:
        self.query = query
        self.passages = passages
