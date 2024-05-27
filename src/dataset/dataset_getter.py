from abc import ABC, abstractmethod
from typing import Dict, List


class DatasetGetter(ABC):
    @abstractmethod
    def get_training_dataset() -> List[Dict[str, object]]:
        pass

    @abstractmethod
    def get_test_dataset() -> List[Dict[str, object]]:
        pass
