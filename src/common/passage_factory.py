from typing import Dict, List
from dataset.dataset_getter import DatasetGetter
from langchain_core.documents import BaseDocumentTransformer


# 3 breakpoint types
# 1. interquartile - 1.5
# 2. percentile - 95
# 3. standard_deviation - 3
class PassageFactory:
    def __init__(
        self, text_splitter: BaseDocumentTransformer, dataset_getter: DatasetGetter
    ) -> None:
        self.text_splitter = text_splitter
        self.dataset_getter = dataset_getter

    def _get_unique_entries(self, dataset: List[Dict[str, object]]):
        unique_rows = {
            row["passage"]: (row["passage"], row["question"], row["answers"])
            for row in dataset
        }
        return set(unique_rows.values())

    def get_passages_for_embedding(self, length=1000) -> List[str]:
        dataset = self.dataset_getter.get_test_dataset()
        unique_entries = self._get_unique_entries(dataset)

        passages = []
        for i, (passage, _, _) in enumerate(unique_entries):
            passages.extend(self.text_splitter.create_documents([passage]))
            print(f"Processed {i+1} passages")
            if i + 1 == length:
                break

        return passages
