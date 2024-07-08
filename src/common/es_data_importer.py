from common.passage_factory import PassageFactory
from repository.es_repository import ESRepository


class ESDataImporter:
    def __init__(
        self,
        repository: ESRepository,
        passage_factory: PassageFactory,
    ):
        self.repository = repository
        self.passage_factory = passage_factory

    def import_data(self):
        passages = self.passage_factory.get_passages()

        for i in range(0, len(passages), 10):
            part_of_passages = passages[i : i + 10]

            self.repository.insert_many(part_of_passages)
            print(f"Processed {i+10} passages")
