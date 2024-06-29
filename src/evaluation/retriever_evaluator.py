from ast import List
import math

from common.passage import Passage
from common.result import Result


class RetrieverEvaluator:
    # Calculate NDCG for top 10 results
    def calculate_ndcg(self, result: Result, correct_passage_title: str) -> float:
        relevances = [
            1 if passage.title == correct_passage_title else 0
            for passage in result.passages
        ]

        sorted_relevances = sorted(relevances, reverse=True)

        dcg = sum((rel / math.log(i + 2, 2)) for i, rel in enumerate(relevances))
        idcg = sum(
            (rel / math.log(i + 2, 2)) for i, rel in enumerate(sorted_relevances)
        )

        return dcg / idcg if idcg != 0 else 0

    # Calculate MRR for top 10 results
    def calculate_mrr(self, result: Result, correct_passage_title: str) -> float:
        for i, passage in enumerate(result.passages):
            if passage.title == correct_passage_title:
                return 1 / (i + 1)
        return 0

    # Calculate recall for top 10 results
    def calculate_recall(
        self, result: Result, correct_passage_title: str, relevant_documents_count: int
    ) -> float:
        relevant_documents = sum(
            1 for passage in result.passages if passage.title == correct_passage_title
        )

        return relevant_documents / relevant_documents_count

    # Calculate accuracy for top 1 result
    def calculate_accuracy(self, result: Result, correct_passage_title: str) -> float:
        return 1 if result.passages[0].title == correct_passage_title else 0
