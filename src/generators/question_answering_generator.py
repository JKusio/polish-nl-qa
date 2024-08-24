from common.passage import Passage
from transformers import pipeline
from generators.generator import Generator


class QuestionAnsweringGenerator(Generator):
    def __init__(self, model_name: str):
        self.model_name = model_name

        self.pipeline = pipeline("question-answering", model=model_name)

    def generate_answer(self, query: str, passages: list[Passage]) -> str:
        context = " ".join([passage.context for passage in passages]).replace("\n", " ")

        return self.pipeline(question=query, context=context)["answer"]
