"""
Curated Dataset Getter - użyj ręcznie wybranych pytań zamiast losowych.

Ten moduł ładuje skurowane datasety (100 najlepszych pytań) i udostępnia
je w tym samym formacie co oryginalne dataset gettery.
"""

import json
import os
from typing import List
from dataclasses import dataclass


@dataclass
class CuratedDatasetEntry:
    """Entry from curated dataset"""

    question: str
    answers: List[str]
    passage_id: str
    context: str = None


class CuratedDatasetGetter:
    """Get curated datasets"""

    # Use absolute path relative to this file
    _BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    CURATED_DIR = os.path.join(_BASE_DIR, "output", "curated_datasets")
    POQUAD_FILE = os.path.join(CURATED_DIR, "curated_poquad_100.json")
    POLQA_FILE = os.path.join(CURATED_DIR, "curated_polqa_100.json")

    @classmethod
    def get_curated_poquad(cls) -> List[CuratedDatasetEntry]:
        """Get curated PoQuAD dataset (100 questions)"""
        return cls._load_curated_dataset(cls.POQUAD_FILE, "PoQuAD")

    @classmethod
    def get_curated_polqa(cls) -> List[CuratedDatasetEntry]:
        """Get curated PolQA dataset (100 questions)"""
        return cls._load_curated_dataset(cls.POLQA_FILE, "PolQA")

    @classmethod
    def _load_curated_dataset(
        cls, filepath: str, dataset_name: str
    ) -> List[CuratedDatasetEntry]:
        """Load curated dataset from JSON file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Curated {dataset_name} dataset not found at {filepath}. "
                f"Run curate_evaluation_dataset.py first to create it."
            )

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        return [
            CuratedDatasetEntry(
                question=entry["question"],
                answers=entry["answers"],
                passage_id=entry["passage_id"],
                context=entry.get("context"),
            )
            for entry in data
        ]

    @classmethod
    def has_curated_datasets(cls) -> tuple:
        """Check which curated datasets exist"""
        return (os.path.exists(cls.POQUAD_FILE), os.path.exists(cls.POLQA_FILE))
