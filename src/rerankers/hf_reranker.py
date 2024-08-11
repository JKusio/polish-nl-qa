import json
from typing import List

from cache.cache import Cache
from common.models_dimensions import RERANKER_MODEL_DIMENSIONS_MAP
from common.passage import Passage
from common.result import Result
from common.utils import get_reranker_hash
from rerankers.reranker import Reranker
from sentence_transformers import CrossEncoder


class HFReranker(Reranker):
    def __init__(self, model_name: str, cache: Cache):
        self.model_name = model_name
        self.model = CrossEncoder(
            model_name, max_length=RERANKER_MODEL_DIMENSIONS_MAP[model_name]
        )
        self.cache = cache

        print(f"Vectorizer with model {model_name} initialized")

    def rerank(self, result: Result, count: int, dataset_key: str) -> Result:
        if (len(result.passages)) == 0:
            return result

        passages_ids = [passage[0].id for passage in result.passages]
        sorted_passages_ids = sorted(passages_ids)

        hash_key = get_reranker_hash(
            self.model_name, result.query, sorted_passages_ids, dataset_key, count
        )

        maybe_cached_result = self.cache.get(hash_key)

        if maybe_cached_result:
            dicts = json.loads(maybe_cached_result)
            passages = [(Passage.from_dict(d["passage"]), d["score"]) for d in dicts]
            return Result(result.query, passages)

        pairs = [[result.query, passage[0].context] for passage in result.passages]

        results = self.model.predict(pairs)
        results_list = results.tolist()

        scored_passages = list(zip(results_list, result.passages))
        sorted_passages = sorted(scored_passages, key=lambda x: x[0], reverse=True)
        top_n_passages = sorted_passages[:count]
        top_n_passages = [(passage, score) for score, (passage, _) in top_n_passages]

        max_score = top_n_passages[0][1]
        min_score = top_n_passages[-1][1]

        score_diff = max_score - min_score

        normalized_passages = [
            (p, 1 if score_diff == 0 else (s - min_score) / score_diff)
            for (p, s) in top_n_passages
        ]

        reranked_result = Result(result.query, normalized_passages)

        result_json = json.dumps(
            [{"passage": p.dict(), "score": s} for (p, s) in normalized_passages]
        )
        self.cache.set(hash_key, result_json)

        return reranked_result
