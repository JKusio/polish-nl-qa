import ast
from typing import List
from datasets import load_dataset
from common.dataset_entry import DatasetEntry
from dataset.dataset_getter import DatasetGetter


class PolqaDatasetGetter(DatasetGetter):
    def __init__(self) -> None:
        self.dataset_name = "ipipan/polqa"

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
                passage,
                self.dataset_name,
                question,
                ast.literal_eval(answers),
                {
                    "question_formulation": question_formulation,
                    "question_type": question_type,
                    "entity_type": entity_type,
                    "passage_id": passage_id,
                },
            )
            for id, title, passage, passage_id, question, answers, relevant, question_formulation, question_type, entity_type in zip(
                dataset["question_id"],
                dataset["passage_title"],
                dataset["passage_wiki"],
                dataset["passage_id"],
                dataset["question"],
                dataset["answers"],
                dataset["relevant"],
                dataset["question_formulation"],
                dataset["question_type"],
                dataset["entity_type"],
            )
            if id and passage and relevant is True
        ]
