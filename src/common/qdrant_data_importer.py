from typing import List
from common.passage_factory import PassageFactory
from repository.qdrant_repository import QdrantRepository
from vectorizer.vectorizer import Vectorizer


class QdrantDataImporter:
    def __init__(
        self,
        repositories: List[QdrantRepository],
        passage_factory: PassageFactory,
        vectorizer: Vectorizer,
    ):
        self.repositories = repositories
        self.passage_factory = passage_factory
        self.vectorizer = vectorizer

    def import_data(self):
        passages = self.passage_factory.get_passages()

        for i in range(0, len(passages), 10):
            passages_and_vectors = [
                (passage, self.vectorizer.get_vector(passage.text))
                for passage in passages[i : i + 10]
            ]

            for repository in self.repositories:
                repository.insert_many_with_vectors(passages_and_vectors)
            print(f"Processed {i+10} passages")
