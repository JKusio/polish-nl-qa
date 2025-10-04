from cache.cache import Cache
from common.passage import Passage
from common.utils import clean_text, get_generator_hash
from generators.generator import Generator
from mlx_lm import load, generate


class InstructionGenerator(Generator):
    def __init__(self, model_name: str, cache: Cache):
        self.model_name = model_name

        model, tokenizer = load(model_name)

        self.model = model
        self.tokenizer = tokenizer

        self.cache = cache

    def generate_single_answer(self, query: str, context: str) -> str:
        prompt = f"""Odpowiedz na pytanie użytkownika wykorzystując wyłącznie informacje z dostarczonych dokumentów. Udziel krótkiej, precyzyjnej odpowiedzi w języku polskim bez dodatkowych komentarzy. Jeżeli w dokumentach nie ma informacji potrzebnych do odpowiedzi, napisz tylko: "Nie udało mi się odnaleźć odpowiedzi na pytanie".

        ### Dokumenty:
        {context}

        ### Pytanie: 
        {query}

        ### Odpowiedź (tylko sama odpowiedź bez wyjaśnień):"""

        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=500,
        )

        stripped_response = (
            response.replace("<s>", "")
            .replace("</s>", "")
            .replace("[INST]", "")
            .replace("[/INST]", "")
            .strip()
        )

        return clean_text(stripped_response)

    def generate_answer(self, query: str, passages: list[Passage]) -> str:
        context = " ".join([passage.context for passage in passages]).replace("\n", " ")

        hash_key = get_generator_hash(query, context, "instruction_v3", self.model_name)

        cached_value = self.cache.get(hash_key)

        if cached_value:
            return cached_value

        answer = self.generate_single_answer(query, context)

        self.cache.set(hash_key, answer)

        return answer
