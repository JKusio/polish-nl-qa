from sentence_transformers import CrossEncoder
from cache import cache
from cache.cache import Cache
from common.models_dimensions import RERANKER_MODEL_DIMENSIONS_MAP
from common.names import HALLUCINATION_MODEL, NER_MODEL, RERANKER_MODEL
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
    AutoModelForSequenceClassification,
)
from common.passage import Passage
from common.utils import (
    get_answer_reranker_hash,
    get_halucination_hash,
    get_ner_hash,
    get_query_reranker_hash,
    get_query_to_passages_reranker_hash,
)


class HallucinationEvaluator:
    def __init__(self, cache: Cache):
        self.cache = cache
        ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL)
        ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL)

        self.ner_pipeline = pipeline(
            "ner",
            model=ner_model,
            tokenizer=ner_tokenizer,
            grouped_entities=True,
            device="mps",
        )

        self.hallucination_model = AutoModelForSequenceClassification.from_pretrained(
            HALLUCINATION_MODEL, trust_remote_code=True
        )

        self.reranker_model = CrossEncoder(
            RERANKER_MODEL, max_length=RERANKER_MODEL_DIMENSIONS_MAP[RERANKER_MODEL]
        )

    def calculate_ner_score(self, answer: str, context: str) -> float:
        hash_key = get_ner_hash(answer, context)
        cached_score = self.cache.get(hash_key)

        if cached_score:
            return cached_score

        answer_entities = self.ner_pipeline(answer)
        context_entities = self.ner_pipeline(context)

        answer_words = [entity["word"].lower() for entity in answer_entities]
        context_words = [entity["word"].lower() for entity in context_entities]

        common_words = set(answer_words) & set(context_words)

        score = float(
            len(common_words) / len(answer_words) if len(answer_words) > 0 else 0
        )

        self.cache.set(hash_key, score)

        return score

    def calculate_hallucination_score(self, answer: str, context: str) -> float:
        hash_key = get_halucination_hash(answer, context)
        cached_score = self.cache.get(hash_key)

        if cached_score:
            return cached_score

        pairs = [[context, answer]]
        results = self.hallucination_model.predict(pairs)

        score = float(results[0].item())
        self.cache.set(hash_key, score)

        return score

    def calculate_answer_reranker_score(
        self, answer: str, passages: list[Passage]
    ) -> float:
        hash_key = get_answer_reranker_hash(answer, passages)
        cached_score = self.cache.get(hash_key)

        if cached_score:
            return cached_score

        pairs = [[answer, passage.context] for passage in passages]
        results = self.reranker_model.predict(pairs)

        score = float(max(results))
        self.cache.set(hash_key, score)

        return score

    def calculate_query_reranker_score(self, query: str, answer: str) -> float:
        hash_key = get_query_reranker_hash(query, answer)
        cached_score = self.cache.get(hash_key)

        if cached_score:
            return cached_score

        pairs = [[query, answer]]
        results = self.reranker_model.predict(pairs)

        score = float(max(results))
        self.cache.set(hash_key, score)

        return score

    def calculate_query_to_passages_score(
        self, query: str, passages: list[Passage]
    ) -> float:
        hash_key = get_query_to_passages_reranker_hash(query, passages)
        cached_score = self.cache.get(hash_key)

        if cached_score:
            return cached_score

        pairs = [[query, passage.context] for passage in passages]
        results = self.reranker_model.predict(pairs)

        score = float(max(results))
        self.cache.set(hash_key, score)

        return score

    def calculate(self, query: str, answer: str, passages: list[Passage]) -> float:
        context = " ".join([passage.context for passage in passages]).replace("\n", " ")

        ner_score = self.calculate_ner_score(answer, context)
        hallucination_score = self.calculate_hallucination_score(answer, context)
        answer_reranker_score = self.calculate_answer_reranker_score(answer, passages)
        query_reranker_score = self.calculate_query_reranker_score(query, answer)
        query_to_passages_score = self.calculate_query_to_passages_score(
            query, passages
        )

        return (
            ner_score * 0.2
            + hallucination_score * 0.5
            + answer_reranker_score * 0.10
            + query_reranker_score * 0.10
            + query_to_passages_score * 0.10
        )
