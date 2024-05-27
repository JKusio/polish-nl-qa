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
                "passage": passage,
                "question": question,
                "answers": answers["text"],
            }
            for passage, question, answers in zip(
                dataset["context"], dataset["question"], dataset["answers"]
            )
        ]

    def get_test_dataset(self) -> List[Dict[str, object]]:
        dataset = load_dataset(self.dataset, split="validation", trust_remote_code=True)
        return [
            {
                "passage": passage,
                "question": question,
                "answers": answers["text"],
            }
            for passage, question, answers in zip(
                dataset["context"], dataset["question"], dataset["answers"]
            )
        ]
