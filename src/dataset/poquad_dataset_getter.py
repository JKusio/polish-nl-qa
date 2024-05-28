from typing import Dict, List
from datasets import load_dataset
from dataset.dataset_getter import DatasetGetter


class PoquadDatasetGetter(DatasetGetter):
    def __init__(self) -> None:
        self.dataset = "clarin-pl/poquad"

    def get_training_dataset(self) -> List[Dict[str, object]]:
        dataset = load_dataset(self.dataset, split="train", trust_remote_code=True)
        return [
            {
                "id": id,
                "passage": passage,
                "question": question,
                "answers": answers["text"],
                "title": title,
            }
            for id, passage, question, answers, title in zip(
                dataset["id"],
                dataset["context"],
                dataset["question"],
                dataset["answers"],
                dataset["title"],
            )
        ]

    def get_test_dataset(self) -> List[Dict[str, object]]:
        dataset = load_dataset(self.dataset, split="validation", trust_remote_code=True)
        return [
            {
                "id": id,
                "passage": passage,
                "question": question,
                "answers": answers["text"],
                "title": title,
            }
            for id, passage, question, answers, title in zip(
                dataset["id"],
                dataset["context"],
                dataset["question"],
                dataset["answers"],
                dataset["title"],
            )
        ]
