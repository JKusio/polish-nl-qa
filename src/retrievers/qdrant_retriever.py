from typing import List
from repository.repository import Repository
from retrievers.retriever import Retriever


class QdrantRetriever(Retriever):
    def __init__(self, repository: Repository):
        self.repository = repository

    def get_relevant_passages(self, query: str) -> List[str]:
        passages = self.repository.find(query)

        return list(map(lambda passage: passage.text, passages))
