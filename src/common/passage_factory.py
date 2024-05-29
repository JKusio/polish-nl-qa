from typing import Dict, List
from common.passage import Passage
from dataset.dataset_getter import DatasetGetter
from langchain_core.documents import BaseDocumentTransformer


# 3 breakpoint types
# 1. interquartile - 1.5
# 2. percentile - 95
# 3. standard_deviation - 3
class PassageFactory:
    def __init__(
        self,
        text_splitter: BaseDocumentTransformer,
        dataset_getter: DatasetGetter,
        chunk_overlap: int = 0,
    ) -> None:
        self.text_splitter = text_splitter
        self.dataset_getter = dataset_getter
        self.chunk_overlap = chunk_overlap

    def _get_unique_entries(
        self, dataset: List[Dict[str, object]]
    ) -> List[Dict[str, object]]:
        unique_rows = {
            row["passage"]: (
                row["id"],
                row["passage"],
                row["title"],
            )
            for row in dataset
        }

        return set(unique_rows.values())

    def get_passages(self, length=1000) -> List[Passage]:
        dataset = self.dataset_getter.get_test_dataset()
        unique_entries = self._get_unique_entries(dataset)

        passages = []
        for i, (id, passage, title) in enumerate(unique_entries):
            splits = self.text_splitter.create_documents([passage])
            start_index = 0
            for split in splits:
                passages.append(Passage(id, split.page_content, title, start_index))
                start_index += len(split.page_content) - self.chunk_overlap
            print(f"Processed {i+1} passages")
            if i + 1 == length:
                break

        return passages
