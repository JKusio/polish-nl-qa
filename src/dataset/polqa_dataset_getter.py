import ast
from typing import Dict, List
from datasets import load_dataset
from dataset.dataset_getter import DatasetGetter


class PolqaDatasetGetter(DatasetGetter):
    def __init__(self) -> None:
        self.dataset = "ipipan/polqa"

    def get_training_dataset(self) -> List[Dict[str, object]]:
        dataset = load_dataset(self.dataset, split="train", trust_remote_code=True)
        return self._convert_to_dict(dataset)

    def get_test_dataset(self) -> List[Dict[str, object]]:
        dataset = load_dataset(self.dataset, split="validation", trust_remote_code=True)
        return self._convert_to_dict(dataset)

    def _convert_to_dict(self, dataset) -> List[Dict[str, str]]:
        return [
            {
                "id": id,
                "passage": passage,
                "question": question,
                "answers": ast.literal_eval(answers),
                "title": title,
            }
            for id, passage, question, answers, title in zip(
                dataset["passage_id"],
                dataset["passage_wiki"],
                dataset["question"],
                dataset["answers"],
                dataset["passage_title"],
            )
            if id and passage
        ]
