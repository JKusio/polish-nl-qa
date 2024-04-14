from vectorizer.vectorizer import Vectorizer
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer


class HFVectorizer(Vectorizer):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name) 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_seq_length = self.model.max_seq_length

        print(f"Vectorizer with model {model_name} initialized")

    def get_vector(self, text: str):
        tokens = self.tokenizer(text, padding="max_length", max_length=self.max_seq_length, truncation=True, return_tensors="pt")
        return self.model.encode(tokens["input_ids"]).tolist()