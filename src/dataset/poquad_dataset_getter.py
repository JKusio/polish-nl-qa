import random
from typing import List
from datasets import load_dataset
from common.dataset_entry import DatasetEntry
from dataset.dataset_getter import DatasetGetter


class PoquadDatasetGetter(DatasetGetter):
    def __init__(self) -> None:
        self.dataset_name = "clarin-pl/poquad"

    def get_unique_questions(self, entries: list[DatasetEntry]) -> List[DatasetEntry]:
        sorted_entries = sorted(entries, key=lambda x: x.id)

        unique_questions = set()
        unique_entries: list[DatasetEntry] = []

        for entry in sorted_entries:
            if entry.question not in unique_questions:
                unique_questions.add(entry.question)
                unique_entries.append(entry)

        return unique_entries

    def get_training_dataset(self) -> List[DatasetEntry]:
        dataset = load_dataset(self.dataset_name, split="train", trust_remote_code=True)
        return self.get_unique_questions(self._convert_from_dict(dataset))

    def get_test_dataset(self) -> List[DatasetEntry]:
        dataset = load_dataset(
            self.dataset_name, split="validation", trust_remote_code=True
        )
        return self.get_unique_questions(self._convert_from_dict(dataset))

    def get_random_n_test(self, n: int, seed: str) -> List[DatasetEntry]:
        dataset = self.get_test_dataset()
        random.seed(seed)
        random_n = random.sample(dataset, n)
        return random_n

    def _convert_from_dict(self, dataset) -> List[DatasetEntry]:
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
