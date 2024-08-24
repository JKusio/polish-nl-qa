from common.passage import Passage
from generators.generator import Generator
from mlx_lm import load, generate


class InstructionGenerator(Generator):
    def __init__(self, model_name: str):
        self.model_name = model_name

        model, tokenizer = load(model_name)

        self.model = model
        self.tokenizer = tokenizer

    def generate_answer(self, query: str, passages: list[Passage]) -> str:
        context = " ".join([passage.context for passage in passages]).replace("\n", " ")

        prompt = f"""
        [INST]
        Wygeneruj krótką odpowiedź na pytanie wyłącznie na podstawie poniższego kontekstu:
        {context}

        Pytanie: {query}
        [/INST]
        """

        response = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=200)

        stripped_response = (
            response.replace("<s>", "")
            .replace("</s>", "")
            .replace("[INST]", "")
            .replace("[/INST]", "")
            .strip()
        )

        return stripped_response
