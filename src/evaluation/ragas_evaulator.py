from sentence_transformers import CrossEncoder
from cache.cache import Cache
from common.models_dimensions import RERANKER_MODEL_DIMENSIONS_MAP
from common.result import Result
import nltk
from nltk.tokenize import sent_tokenize
import ssl
from mlx_lm import load, generate
from common.utils import (
    get_answer_relevance_hash,
    get_faithfulness_hash,
    get_query_to_context_relevance_hash,
)
from vectorizer.vectorizer import Vectorizer


class RAGASEvaluator:
    def __init__(
        self,
        reranker_model_name: str,
        cache: Cache,
        generator_model_name: str,
        vectorizer: Vectorizer,
    ):
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        nltk.download("punkt")

        self.raranker_model_name = reranker_model_name
        self.raranker_model = CrossEncoder(
            reranker_model_name,
            max_length=RERANKER_MODEL_DIMENSIONS_MAP[reranker_model_name],
        )
        model, tokenizer = load(generator_model_name)
        self.generator_model = model
        self.generator_tokenizer = tokenizer
        self.cache = cache
        self.vectorizer = vectorizer
        pass

    def context_precision(self, result: Result, correct_passage_id: str) -> float:

        precisions_sum = 0

        for i in range(1, len(result.passages) + 1):
            sub_precision = 0

            for j in range(0, i):
                if result.passages[j][0].id == correct_passage_id:
                    sub_precision += 1

            relevant_at_i = (
                1 if result.passages[i - 1][0].id == correct_passage_id else 0
            )

            precisions_sum += (sub_precision / i) * relevant_at_i

        sum_of_relevant_passages = sum(
            [1 for passage in result.passages if passage[0].id == correct_passage_id]
        )

        value = (
            precisions_sum / sum_of_relevant_passages
            if sum_of_relevant_passages > 0
            else 0
        )

        return value

    def context_recall(self, result: Result, correct_passage_id: str) -> float:
        is_any_relevant = False

        for passage in result.passages:
            if passage[0].id == correct_passage_id:
                is_any_relevant = True
                break

        return 1 if is_any_relevant else 0

    def _get_relevancy_between_answer_and_context(
        self, answer: str, contexts: list[str]
    ):
        pairs = [[answer, context] for context in contexts]
        results = self.raranker_model.predict(pairs)
        return 1 if max(results) > 0.5 else 0

    def faithfulness(self, result: Result, answer: str) -> float:
        sentences = sent_tokenize(answer)
        context = " ".join([passage[0].context for passage in result.passages])
        context_sentences = sent_tokenize(context)

        hash_key = get_faithfulness_hash(answer, context)

        maybe_faithfulness = self.cache.get(hash_key)

        if maybe_faithfulness:
            return float(maybe_faithfulness)

        faihtfulness = sum(
            [
                self._get_relevancy_between_answer_and_context(
                    sentence, context_sentences
                )
                for sentence in sentences
            ]
        ) / len(sentences)

        self.cache.set(hash_key, str(faihtfulness))

        return faihtfulness

    def answer_relevance(self, original_question: str, answer: str) -> float:
        hash_key = get_answer_relevance_hash(original_question, answer)

        maybe_answer_relevance = self.cache.get(hash_key)

        if maybe_answer_relevance:
            return float(maybe_answer_relevance)

        prompt = f"""
            [INST]
            Na podstawie podanego kontekstu wygeneruj trzy pytania, które mogłyby zostać zadane w kontekście tego tekstu.
            Zwróć pytania w formacie
            1. Pierwsze pytanie
            2. Drugie pytanie
            3. Trzecie pytanie
            
            Każde pytanie musi być zakończone kropką i znajdować się w osobnej linii. Zwróć tylko pytania z numerami 1, 2 i 3.
            
            Kontekst: {answer.replace("\n", " ")}
            [/INST]
        """

        generated_questions = generate(
            self.generator_model,
            self.generator_tokenizer,
            prompt=prompt,
            max_tokens=300,
        )

        questions = generated_questions.split("\n")
        filtered_questions = [
            q.strip()
            for q in questions
            if q.strip() and len(q.strip()) > 0 and q.strip()[0] in {"1", "2", "3"}
        ][:3]

        original_question_vector = self.vectorizer.get_vector(
            f"zapytanie: {original_question}"
        )
        sentence_vectiors = [
            self.vectorizer.get_vector(f"zapytanie: {q}") for q in filtered_questions
        ]
        results = [
            self.vectorizer.get_similarity(original_question_vector, v).item()
            for v in sentence_vectiors
        ]

        score = (
            (1 / len(filtered_questions)) * sum(results)
            if len(filtered_questions) > 0
            else 0
        )

        self.cache.set(hash_key, str(score))

        return score

    def query_to_context_relevance(self, result: Result) -> float:
        hash_key = get_query_to_context_relevance_hash(result)

        maybe_query_to_context_relevance = self.cache.get(hash_key)

        if maybe_query_to_context_relevance:
            return float(maybe_query_to_context_relevance)

        query_vector = self.vectorizer.get_vector(result.query)
        context_vectors = [
            self.vectorizer.get_vector(passage[0].context)
            for passage in result.passages
        ]

        result = [
            self.vectorizer.get_similarity(query_vector, v).item()
            for v in context_vectors
        ]

        score = (1 / len(context_vectors)) * sum(result)

        self.cache.set(hash_key, str(score))

        return score

    def ragas(self, result: Result, correct_passage_id: str, answer: str) -> float:
        context_precision = self.context_precision(result, correct_passage_id)
        context_recall = self.context_recall(result, correct_passage_id)
        faithfulness = self.faithfulness(result, answer)
        answer_relevance = self.answer_relevance(result.query, answer)

        return (
            context_precision + context_recall + faithfulness + answer_relevance
        ) / 4

    def hallucination(self, result: Result, answer: str) -> float:
        faithfulness = self.faithfulness(result, answer)
        answer_relevance = self.answer_relevance(result.query, answer)
        context_relevance = self.query_to_context_relevance(result)

        return (faithfulness + answer_relevance + context_relevance) / 3
