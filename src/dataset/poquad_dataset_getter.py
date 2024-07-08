from typing import List
from datasets import load_dataset
from common.dataset_entry import DatasetEntry
from dataset.dataset_getter import DatasetGetter


class PoquadDatasetGetter(DatasetGetter):
    def __init__(self) -> None:
        self.dataset_name = "clarin-pl/poquad"

    def get_training_dataset(self) -> List[DatasetEntry]:
        dataset = load_dataset(self.dataset_name, split="train", trust_remote_code=True)
        return self._convert_to_dict(dataset)

    def get_test_dataset(self) -> List[DatasetEntry]:
        dataset = load_dataset(
            self.dataset_name, split="validation", trust_remote_code=True
        )
        return self._convert_to_dict(dataset)

    def _convert_to_dict(self, dataset) -> List[DatasetEntry]:
        return [
            DatasetEntry(
                id,
                title,
                context,
                self.dataset_name,
                question,
                answers["text"],
                {"answer_start": answers["answer_start"][0]},
            )
            for id, context, question, answers, title in zip(
                dataset["id"],
                dataset["context"],
                dataset["question"],
                dataset["answers"],
                dataset["title"],
            )
        ]
