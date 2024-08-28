from sklearn.metrics import f1_score
from common.passage import Passage
from common.result import Result


class QAEvaluator:
    def calculate_em(self, answer: str, correct_answer) -> float:
        unified_answer = answer.lower().strip()
        unified_correct_answer = correct_answer.lower().strip()

        return 1 if unified_answer == unified_correct_answer else 0

    def calculate_f1_score(self, answer: str, correct_answer) -> float:
        unified_answer = answer.lower().strip()
        unified_correct_answer = correct_answer.lower().strip()

        answer_tokens = unified_answer.split()
        correct_answer_tokens = unified_correct_answer.split()

        common_tokens = 0
        correct_answer_token_counts = {}

        for token in correct_answer_tokens:
            if token in correct_answer_token_counts:
                correct_answer_token_counts[token] += 1
            else:
                correct_answer_token_counts[token] = 1

        for token in answer_tokens:
            if (
                token in correct_answer_token_counts
                and correct_answer_token_counts[token] > 0
            ):
                common_tokens += 1
                correct_answer_token_counts[token] -= 1

        precision = common_tokens / len(answer_tokens) if len(answer_tokens) > 0 else 0

        recall = (
            common_tokens / len(correct_answer_tokens)
            if len(correct_answer_tokens) > 0
            else 0
        )

        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if precision + recall > 0
            else 0
        )

        return f1_score
