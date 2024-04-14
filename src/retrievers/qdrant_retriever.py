from repository.repository import Repository
from retrievers.retriever import Retriever

class QdrantRetriever(Retriever):
    def __init__(self, repository: Repository):
        self.repository = repository

    def get_relevant_docs(self, query: str):
        return self.repository.find(query)