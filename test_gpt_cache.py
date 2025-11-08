#!/usr/bin/env python3
"""
Test if GPT answers are in cache
"""
import sys

sys.path.insert(0, "src")

import json
from cache.cache import Cache
from common.utils import get_generator_hash, replace_slash_with_dash

cache = Cache()

# Load GPT answers
print("Loading GPT answers to cache...")
datasets = [
    "clarin-pl-poquad-500",
    "clarin-pl-poquad-1000",
    "clarin-pl-poquad-2000",
    "clarin-pl-poquad-100000",
    "ipipan-polqa-500",
    "ipipan-polqa-1000",
    "ipipan-polqa-2000",
    "ipipan-polqa-100000",
]

loaded_count = 0
for dataset_key in datasets:
    filename = replace_slash_with_dash(f"gpt-4o-mini_{dataset_key}.jsonl")
    filepath = f"openai_batches/{filename}"

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            custom_id = data["custom_id"]
            answer = data["response"]["body"]["choices"][0]["message"]["content"]
            cache.set(custom_id, answer)
            loaded_count += 1
    print(f"✓ Loaded {filename}")

print(f"\n✅ Loaded {loaded_count} answers to cache")

# Test retrieval
print("\n" + "=" * 80)
print("TESTING CACHE RETRIEVAL")
print("=" * 80)

# Take first answer from poquad-500 file
with open("openai_batches/gpt-4o-mini_clarin-pl-poquad-500.jsonl", "r") as f:
    first_line = f.readline()
    data = json.loads(first_line)
    test_key = data["custom_id"]
    expected_answer = data["response"]["body"]["choices"][0]["message"]["content"]

print(f"\nTest key: {test_key[:80]}...")
print(f"Expected answer: {expected_answer[:100]}...")

cached_answer = cache.get(test_key)
print(f"Cached answer: {cached_answer[:100] if cached_answer else 'NOT FOUND'}...")

if cached_answer == expected_answer:
    print("\n✅ SUCCESS! Cache is working correctly!")
else:
    print("\n❌ FAILED! Cache mismatch!")
