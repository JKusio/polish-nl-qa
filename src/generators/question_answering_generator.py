from cache.cache import Cache
from common.passage import Passage
from transformers import pipeline, AutoTokenizer
from common.utils import (
    clean_text,
    get_generator_hash,
    split_into_chunks,
    split_text_into_token_chunks,
)
from generators.generator import Generator

MAX_LENGTH = 512


class QuestionAnsweringGenerator(Generator):
    def __init__(self, model_name: str, cache: Cache):
        self.model_name = model_name

        self.pipeline = pipeline(
            "question-answering",
            model=model_name,
            device="mps",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.cache = cache

    def generate_answer(self, query: str, passages: list[Passage]) -> str:
        context = " ".join([passage.context for passage in passages]).replace("\n", " ")

        hash_key = get_generator_hash(query, context, "question_answering")

        cached_value = self.cache.get(hash_key)

        if cached_value:
            return cached_value

        chunks = split_text_into_token_chunks(context, self.tokenizer, MAX_LENGTH, 64)

        all_answers = []

        for chunk in chunks:
            result = self.pipeline(question=query, context=chunk)
            all_answers.append(result)

        best_answer = max(all_answers, key=lambda x: x["score"])

        return clean_text(best_answer["answer"])
