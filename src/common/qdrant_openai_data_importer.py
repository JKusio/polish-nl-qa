from common.passage_factory import PassageFactory
from common.utils import get_query_with_prefix
from repository.qdrant_repository import QdrantRepository
from vectorizer.vectorizer import Vectorizer


class QdrantOpenAIDataImporter:
    def __init__(
        self,
        repository: QdrantRepository,
        passage_factory: PassageFactory,
    ):
        self.repository = repository
        self.passage_factory = passage_factory

    def import_data(self, batch):
        passages = self.passage_factory.get_passages()

        for i in range(0, len(passages), 10):
            passages_and_vectors = [
                (
                    passage,
                    next(
                        (
                            obj
                            for obj in batch
                            if obj["custom_id"] == f"{passage.id}-{passage.start_index}"
                        ),
                        None,
                    )["response"]["body"]["data"][0]["embedding"],
                )
                for passage in passages[i : i + 10]
            ]

            self.repository.insert_many_with_vectors(passages_and_vectors)
            print(f"Processed {i+10} passages")
