from typing import Dict, List
from common.dataset_entry import DatasetEntry
from common.passage import Passage
from common.utils import get_dataset_key
from dataset.dataset_getter import DatasetGetter
from langchain_core.documents import BaseDocumentTransformer


class PassageFactory:
    def __init__(
        self,
        text_splitter: BaseDocumentTransformer,
        dataset_getter: DatasetGetter,
    ) -> None:
        self.text_splitter = text_splitter
        self.dataset_getter = dataset_getter
        self.chunk_size = self.text_splitter._chunk_size
        self.chunk_overlap = self.text_splitter._chunk_overlap

    def _get_unique_entries(self, dataset: List[DatasetEntry]) -> List[DatasetEntry]:
        unique_entries: List[DatasetEntry] = []
        for entry in dataset:
            if not any(
                entry.context == unique_entry.context
                and entry.title == unique_entry.title
                for unique_entry in unique_entries
            ):
                unique_entries.append(entry)

        return unique_entries

    def get_passages(self, length=1000) -> List[Passage]:
        dataset = self.dataset_getter.get_test_dataset()
        unique_entries = self._get_unique_entries(dataset)

        passages: List[Passage] = []
        for i, (entry) in enumerate(unique_entries):
            splits = self.text_splitter.create_documents([entry.context])
            start_index = 0
            for split in splits:
                passages.append(
                    Passage(
                        entry.id,
                        entry.title,
                        split.page_content,
                        start_index,
                        entry.dataset,
                        get_dataset_key(entry.dataset, self.chunk_size),
                        entry.metadata,
                    )
                )
                start_index += len(split.page_content) - self.chunk_overlap
            print(f"Processed {i+1} passages")
            if i + 1 == length:
                break

        return passages
