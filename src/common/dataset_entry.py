import hashlib
from typing import List


class DatasetEntry:
    def __init__(
        self,
        id: str,
        title: str,
        context: str,
        dataset: str,
        question: str,
        answers: List[str],
        metadata: dict = {},
    ):
        self.id = id
        self.passage_id = hashlib.sha256((title + context).encode()).hexdigest()
        self.title = title
        self.context = context
        self.dataset = dataset
        self.question = question
        self.answers = answers
        self.metadata = metadata

    def __str__(self) -> str:
        return f"DatasetEntry(id={self.id}, passage_id={self.passage_id}, title={self.title}, context={self.context}, dataset={self.dataset} question={self.question}, answers={self.answers}, metadata={self.metadata})"
