from regex import P
from sentence_transformers import CrossEncoder
from common.models_dimensions import RERANKER_MODEL_DIMENSIONS_MAP
from common.names import HALLUCINATION_MODEL, NER_MODEL, RERANKER_MODEL
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
    AutoModelForSequenceClassification,
)

from common.passage import Passage


class HallucinationEvaluator:
    def __init__(self):
        ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL)
        ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL)

        self.ner_pipeline = pipeline(
            "ner", model=ner_model, tokenizer=ner_tokenizer, grouped_entities=True
        )

        self.hallucination_model = AutoModelForSequenceClassification.from_pretrained(
            HALLUCINATION_MODEL, trust_remote_code=True
        )

        self.reranker_model = CrossEncoder(
            RERANKER_MODEL, max_length=RERANKER_MODEL_DIMENSIONS_MAP[RERANKER_MODEL]
        )

    def calculate_ner_score(self, answer: str, context: str) -> float:
        answer_entities = self.ner_pipeline(answer)
        context_entities = self.ner_pipeline(context)

        answer_words = [entity["word"].lower() for entity in answer_entities]
        context_words = [entity["word"].lower() for entity in context_entities]

        common_words = set(answer_words) & set(context_words)

        return len(common_words) / len(answer_words) if len(answer_words) > 0 else 0

    def calculate_hallucination_score(self, answer: str, context: str) -> float:
        pairs = [[context, answer]]
        results = self.hallucination_model.predict(pairs)

        return results[0].item()

    def calculate_reranker_score(self, answer: str, passages: list[Passage]) -> float:
        pairs = [[answer, passage.context] for passage in passages]
        results = self.reranker_model.predict(pairs)
        return max(results)

    def calculate(self, answer: str, passages: list[Passage]) -> float:
        context = " ".join([passage.context for passage in passages]).replace("\n", " ")

        ner_score = self.calculate_ner_score(answer, context)
        hallucination_score = self.calculate_hallucination_score(answer, context)
        reranker_score = self.calculate_reranker_score(answer, passages)

        return ner_score * 0.33 + hallucination_score * 0.33 + reranker_score * 0.33
