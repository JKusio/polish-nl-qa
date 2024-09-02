import json
from typing import Any

import torch
from cache.cache import Cache
from common.utils import get_vectorizer_hash
from vectorizer.vectorizer import Vectorizer
from sentence_transformers import SentenceTransformer


class HFVectorizer(Vectorizer):
    def __init__(self, model_name: str, cache: Cache):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.max_seq_length = self.model.max_seq_length
        self.cache = cache

        print(f"Vectorizer with model {model_name} initialized")

    def get_vector(self, query: str) -> Any:
        hash_key = get_vectorizer_hash(self.model_name, query)

        maybe_hashed_vector = self.cache.get(hash_key)

        if maybe_hashed_vector:
            vector_json = json.loads(maybe_hashed_vector)
            hashed_vector = torch.tensor(vector_json)
            return hashed_vector

        hashed_vector = self.model.encode(query, convert_to_tensor=True)
        vector_list = hashed_vector.tolist()
        vector_json = json.dumps(vector_list)
        self.cache.set(hash_key, vector_json)

        return hashed_vector

    def get_similarity(self, vector1: Any, vector2: Any) -> float:
        return self.model.similarity(vector1, vector2)
