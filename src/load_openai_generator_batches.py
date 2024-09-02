import json
from cache.cache import Cache
from common.utils import get_all_openai_model_combinations, replace_slash_with_dash

def main():
    cache = Cache()
    
    get_batch_file(cache)

def get_batch_file(cache: Cache):
    combinations = get_all_openai_model_combinations()
    
    for _, _, dataset_key in combinations:
        filename = replace_slash_with_dash(
            f"{"gpt-4o-mini"}_{dataset_key}.jsonl"
        )
            
        batch_data = []
        with open(
            f"openai_batches/{filename}",
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                batch_data.append(json.loads(line))
    

        for data in batch_data:
            cache.set(data["custom_id"], data["response"]["body"]["choices"][0]["message"]["content"])

main()
