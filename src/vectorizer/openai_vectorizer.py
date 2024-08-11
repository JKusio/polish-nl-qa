import json
from typing import Any
import torch
from cache.cache import Cache
from common.utils import get_vectorizer_hash
from vectorizer.vectorizer import Vectorizer


class OpenAIVectorizer(Vectorizer):
    def __init__(self, model_name: str, cache: Cache):
        self.model_name = model_name
        self.cache = cache

        print(f"Vectorizer with model {model_name} initialized")

    def get_vector(self, query: str) -> Any:
        hash_key = get_vectorizer_hash(self.model_name, query)

        maybe_hashed_vector = self.cache.get(hash_key)

        if maybe_hashed_vector:
            vector_json = json.loads(maybe_hashed_vector)
            hashed_vector = torch.tensor(vector_json)
            return hashed_vector

        return []
