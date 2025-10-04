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
import numpy as np


class RAGASEvaluatorV2:
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

        nltk.download("punkt", quiet=True)

        self.reranker_model_name = reranker_model_name
        self.reranker_model = CrossEncoder(
            reranker_model_name,
            max_length=RERANKER_MODEL_DIMENSIONS_MAP[reranker_model_name],
        )
        model, tokenizer = load(generator_model_name)
        self.generator_model = model
        self.generator_tokenizer = tokenizer
        self.cache = cache
        self.vectorizer = vectorizer

    def context_precision(self, result: Result, correct_passage_id: str) -> float:
        """Unchanged - measures if relevant passage appears early"""
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
        """Unchanged - binary check if correct passage is retrieved"""
        is_any_relevant = False

        for passage in result.passages:
            if passage[0].id == correct_passage_id:
                is_any_relevant = True
                break

        return 1 if is_any_relevant else 0

    def faithfulness(self, result: Result, answer: str) -> float:
        """
        IMPROVED: Uses continuous scores instead of binary threshold
        Measures how well each answer sentence is grounded in the context
        """
        # Handle edge cases
        if not answer or not answer.strip():
            return 0.0

        sentences = sent_tokenize(answer)
        if not sentences:
            return 0.0

        context = " ".join([passage[0].context for passage in result.passages])
        context_sentences = sent_tokenize(context)

        if not context_sentences:
            return 0.0

        hash_key = get_faithfulness_hash(answer, context)
        maybe_faithfulness = self.cache.get(hash_key)

        if maybe_faithfulness:
            return float(maybe_faithfulness)

        # For each answer sentence, find max similarity with context sentences
        faithfulness_scores = []
        for sentence in sentences:
            if not sentence.strip():
                continue

            pairs = [[sentence, context_sent] for context_sent in context_sentences]
            scores = self.reranker_model.predict(pairs)

            # Use max score (best match) and apply sigmoid-like transformation
            max_score = float(np.max(scores))
            # Transform score to 0-1 range with more gradual transition
            # Scores > 0 become > 0.5, scores < 0 become < 0.5
            normalized_score = 1 / (1 + np.exp(-5 * max_score))
            faithfulness_scores.append(normalized_score)

        faithfulness = np.mean(faithfulness_scores) if faithfulness_scores else 0.0
        self.cache.set(hash_key, str(faithfulness))

        return float(faithfulness)

    def answer_relevance(self, original_question: str, answer: str) -> float:
        """
        IMPROVED: Better prompt and more robust parsing
        Generates questions from answer and compares to original
        """
        # Handle edge cases
        if not answer or not answer.strip():
            return 0.0

        hash_key = get_answer_relevance_hash(original_question, answer)
        maybe_answer_relevance = self.cache.get(hash_key)

        if maybe_answer_relevance:
            return float(maybe_answer_relevance)

        # Improved prompt with better instructions
        prompt = f"""Wygeneruj dokładnie 3 pytania, na które podany tekst mógłby być odpowiedzią. Pytania powinny być konkretne i istotne.

Tekst: {answer.replace("\n", " ")}

Wygeneruj pytania w formacie:
1. [pytanie]
2. [pytanie]
3. [pytanie]

Pytania:"""

        generated_questions = generate(
            self.generator_model,
            self.generator_tokenizer,
            prompt=prompt,
            max_tokens=300,
        )

        # More robust parsing
        questions = []
        for line in generated_questions.split("\n"):
            line = line.strip()
            # Look for lines starting with 1., 2., or 3.
            if line and len(line) > 3:
                if (
                    line.startswith("1.")
                    or line.startswith("2.")
                    or line.startswith("3.")
                ):
                    # Remove the number prefix
                    question = line[2:].strip()
                    if len(question) > 5:  # Minimum length filter
                        questions.append(question)

        # If we couldn't parse 3 questions, try alternative format
        if len(questions) < 3:
            questions = []
            lines = [l.strip() for l in generated_questions.split("\n") if l.strip()]
            for line in lines[:3]:  # Take first 3 non-empty lines
                # Remove common prefixes
                for prefix in ["1.", "2.", "3.", "1)", "2)", "3)", "-", "*"]:
                    if line.startswith(prefix):
                        line = line[len(prefix) :].strip()
                if len(line) > 5:
                    questions.append(line)

        if not questions:
            return 0.0

        # Calculate similarity between original and generated questions
        original_question_vector = self.vectorizer.get_vector(
            f"zapytanie: {original_question}"
        )

        similarities = []
        for q in questions[:3]:  # Use at most 3 questions
            q_vector = self.vectorizer.get_vector(f"zapytanie: {q}")
            similarity = self.vectorizer.get_similarity(
                original_question_vector, q_vector
            ).item()
            similarities.append(similarity)

        score = np.mean(similarities) if similarities else 0.0
        self.cache.set(hash_key, str(score))

        return float(score)

    def query_to_context_relevance(self, result: Result) -> float:
        """
        IMPROVED: Uses weighted average based on passage position
        Earlier passages (more relevant) get higher weight
        """
        hash_key = get_query_to_context_relevance_hash(result)
        maybe_query_to_context_relevance = self.cache.get(hash_key)

        if maybe_query_to_context_relevance:
            return float(maybe_query_to_context_relevance)

        if not result.passages:
            return 0.0

        query_vector = self.vectorizer.get_vector(f"zapytanie: {result.query}")

        # Calculate similarities with position-based weights
        weighted_scores = []
        total_weight = 0

        for i, passage in enumerate(result.passages):
            context_vector = self.vectorizer.get_vector(passage[0].context)
            similarity = self.vectorizer.get_similarity(
                query_vector, context_vector
            ).item()

            # Higher weight for earlier passages (exponential decay)
            weight = 1.0 / (i + 1)  # 1.0, 0.5, 0.33, 0.25, 0.2, ...
            weighted_scores.append(similarity * weight)
            total_weight += weight

        score = sum(weighted_scores) / total_weight if total_weight > 0 else 0.0
        self.cache.set(hash_key, str(score))

        return float(score)

    def answer_correctness(self, answer: str, correct_answers: list[str]) -> float:
        """
        NEW: Direct comparison between generated answer and ground truth
        Uses both semantic similarity and keyword overlap
        """
        if not answer or not answer.strip() or not correct_answers:
            return 0.0

        # Normalize text
        answer_lower = answer.lower().strip()

        # Semantic similarity using vectorizer
        answer_vector = self.vectorizer.get_vector(answer)

        semantic_scores = []
        keyword_scores = []

        for correct_answer in correct_answers:
            correct_lower = correct_answer.lower().strip()

            # Semantic similarity
            correct_vector = self.vectorizer.get_vector(correct_answer)
            semantic_sim = self.vectorizer.get_similarity(
                answer_vector, correct_vector
            ).item()
            semantic_scores.append(semantic_sim)

            # Keyword overlap (Jaccard similarity)
            answer_words = set(answer_lower.split())
            correct_words = set(correct_lower.split())

            if answer_words and correct_words:
                intersection = len(answer_words & correct_words)
                union = len(answer_words | correct_words)
                keyword_sim = intersection / union if union > 0 else 0
                keyword_scores.append(keyword_sim)
            else:
                keyword_scores.append(0.0)

        # Take max score across all correct answers
        max_semantic = max(semantic_scores) if semantic_scores else 0.0
        max_keyword = max(keyword_scores) if keyword_scores else 0.0

        # Combine semantic and keyword scores (weighted average)
        score = 0.7 * max_semantic + 0.3 * max_keyword

        return float(score)

    def ragas(
        self,
        result: Result,
        correct_passage_id: str,
        answer: str,
        correct_answers: list[str] = None,
    ) -> float:
        """
        IMPROVED: Better weighted combination of metrics with answer correctness
        """
        context_precision = self.context_precision(result, correct_passage_id)
        context_recall = self.context_recall(result, correct_passage_id)
        faithfulness = self.faithfulness(result, answer)
        answer_relevance = self.answer_relevance(result.query, answer)

        # Base score from original 4 metrics with better weighting
        # Faithfulness and answer_relevance are most important for answer quality
        retrieval_score = (context_precision + context_recall) / 2  # 0-1
        answer_quality_score = (faithfulness + answer_relevance) / 2  # 0-1

        # Combine: 30% retrieval, 70% answer quality
        base_score = 0.3 * retrieval_score + 0.7 * answer_quality_score

        # If we have ground truth answers, incorporate correctness
        if correct_answers:
            correctness = self.answer_correctness(answer, correct_answers)
            # Final score: 60% base metrics, 40% correctness
            final_score = 0.6 * base_score + 0.4 * correctness
        else:
            final_score = base_score

        return float(final_score)

    def hallucination(self, result: Result, answer: str) -> float:
        """
        IMPROVED: Better detection of hallucinations
        Focuses on faithfulness to context and answer coherence
        """
        faithfulness = self.faithfulness(result, answer)
        answer_relevance = self.answer_relevance(result.query, answer)
        context_relevance = self.query_to_context_relevance(result)

        # Faithfulness is most important for hallucination detection
        # If answer is faithful to context, it's less likely to be hallucination
        hallucination_score = (
            0.6 * faithfulness  # Most important
            + 0.25 * answer_relevance  # Is answer relevant to question?
            + 0.15 * context_relevance  # Is context relevant to question?
        )

        return float(hallucination_score)
