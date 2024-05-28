from common.passage_factory import PassageFactory
from repository.qdrant_repository import QdrantRepository


class QdrantDataImporter:
    def __init__(self, repository: QdrantRepository, passage_factory: PassageFactory):
        self.repository = repository
        self.passage_factory = passage_factory

    def import_data(self):
        passages = self.passage_factory.get_passages_for_embedding()

        for i in range(0, len(passages), 10):
            self.qdrant_repository.insert_many(passages[i : i + 10])
            print(f"Inserted {i + 10} passages")
