from cache.cache import Cache
from common.passage import Passage
from common.utils import get_generator_hash
from generators.generator import Generator


class OpenAIGenerator(Generator):
    def __init__(self, cache: Cache):
        self.model_name = "gpt-4o-mini"
        self.cache = cache

    def generate_answer(self, query: str, passages: list[Passage]) -> str:
        context = " ".join([passage.context for passage in passages]).replace("\n", " ")

        hash_key = get_generator_hash(query, context, "instruction", self.model_name)

        cached_value = self.cache.get(hash_key)

        if cached_value:
            return cached_value

        print(f"WARNING! GPT answer not found in cache for key: {hash_key[:50]}...")
        print(f"Question: {query[:100]}...")
        print("Make sure to load OpenAI batch results to cache first!")
        return ""
