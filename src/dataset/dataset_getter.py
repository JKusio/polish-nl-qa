from abc import ABC, abstractmethod
from typing import List

from common.dataset_entry import DatasetEntry


class DatasetGetter(ABC):
    @abstractmethod
    def get_training_dataset() -> List[DatasetEntry]:
        pass

    @abstractmethod
    def get_test_dataset() -> List[DatasetEntry]:
        pass
