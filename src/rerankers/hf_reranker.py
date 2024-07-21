import json
from typing import List
from cache.cache import Cache
from common.passage import Passage
from common.utils import get_reranker_hash
from rerankers.reranker import Reranker
from sentence_transformers import CrossEncoder


class HFReranker(Reranker):
    def __init__(self, model_name: str, cache: Cache):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)
        self.cache = cache

        print(f"Vectorizer with model {model_name} initialized")

    def get_relevant_passages(
        self, query: str, passages: List[Passage], count: int
    ) -> List[Passage]:
        passages_ids = list(map(lambda passage: passage.id, passages))
        sorted_passages_ids = sorted(passages_ids)

        reranker_hash = get_reranker_hash(
            self.model_name, query, sorted_passages_ids, count
        )

        maybe_cached_result = self.cache.get(reranker_hash)

        if maybe_cached_result:
            json_result = json.loads(maybe_cached_result)
            return list(map(lambda passage: Passage.from_dict(passage), json_result))

        pairs = [
            [query, passage]
            for passage in list(map(lambda passage: passage.context, passages))
        ]

        results = self.model.predict(pairs)
        results_list = results.tolist()

        scored_passages = list(zip(results_list, passages))
        sorted_passages = sorted(scored_passages, key=lambda x: x[0], reverse=True)
        top_n_passages = sorted_passages[:count]
        top_n_passages = [passage for _, passage in top_n_passages]

        top_n_passages_dict = list(map(lambda passage: passage.dict(), top_n_passages))

        self.cache.set(reranker_hash, json.dumps(top_n_passages_dict))

        return top_n_passages
