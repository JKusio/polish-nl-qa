from sqlalchemy import all_
from cache.cache import Cache
from common.passage import Passage
from common.utils import clean_text, get_generator_hash, split_text_into_token_chunks
from generators.generator import Generator
from mlx_lm import load, generate


class InstructionGenerator(Generator):
    def __init__(self, model_name: str, cache: Cache):
        self.model_name = model_name

        model, tokenizer = load(model_name)

        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = 100000

        self.cache = cache

    def chunk_and_generate(self, query: str, context: str, i=0) -> str:
        chunks = split_text_into_token_chunks(
            context, self.tokenizer, self.max_tokens, 64
        )

        all_answers = []

        for chunk in chunks:
            prompt = f"""
            Odpowiedz na pytanie użytkownika wykorzystując tylko informacje znajdujące się w dokumentach, a nie wcześniejszą wiedzę. Udziel wysokiej jakości, poprawnej gramatycznie odpowiedzi w języku polskim. Jeżeli w dokumentach nie ma informacji potrzebnych do odpowiedzi na pytanie, zamiast odpowiedzi zwróć tekst: "Nie udało mi się odnaleźć odpowiedzi na pytanie".
          
            ### Dokumenty:
            {chunk}
            
            ### Pytanie: 
            {query}
            
            ### Odpowiedź:
            """

            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=300,
            )
            stripped_response = (
                response.replace("<s>", "")
                .replace("</s>", "")
                .replace("[INST]", "")
                .replace("[/INST]", "")
                .strip()
            )
            all_answers.append(stripped_response)

        answer = None

        if i > 3:
            return all_answers[0]

        if len(chunks) == 1:
            answer = all_answers[0]
        else:
            all_answer_context = " ".join([text for text in all_answers]).replace(
                "\n", " "
            )

            answer = self.chunk_and_generate(query, all_answer_context, i + 1)

        return clean_text(answer)

    def generate_answer(self, query: str, passages: list[Passage]) -> str:
        context = " ".join([passage.context for passage in passages]).replace("\n", " ")

        hash_key = get_generator_hash(query, context, "instruction", self.model_name)

        cached_value = self.cache.get(hash_key)

        if cached_value:
            return cached_value

        answer = self.chunk_and_generate(query, context)

        self.cache.set(hash_key, answer)

        return answer
