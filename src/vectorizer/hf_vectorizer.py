from typing import Any
from cache.cache import Cache
from vectorizer.vectorizer import Vectorizer
from sentence_transformers import SentenceTransformer


class HFVectorizer(Vectorizer):
    def __init__(self, model_name: str, cache: Cache):
        self.model = SentenceTransformer(model_name)
        self.max_seq_length = self.model.max_seq_length

        print(f"Vectorizer with model {model_name} initialized")

    def get_vector(self, text: str) -> Any:
        return self.model.encode(text, convert_to_tensor=True)
