#!/usr/bin/env python3
"""
Show passages retrieved for question 10 with OpenAI retriever
"""
import sys

sys.path.insert(0, "src")

from qdrant_client import QdrantClient
from qdrant_client.models import Distance
from cache.cache import Cache
from dataset.curated_dataset_getter import CuratedDatasetGetter
from repository.qdrant_openai_repository import QdrantOpenAIRepository
from retrievers.qdrant_retriever import QdrantRetriever
from common.names import OPENAI_EMBEDDING_MODEL_NAMES

# Initialize
cache = Cache()
qdrant_client = QdrantClient(host="localhost", port=6333)

# Get curated PoQuAD dataset
poquad_dataset = CuratedDatasetGetter.get_curated_poquad()

# Get question 10 (index 9)
entry = poquad_dataset[9]
print("=" * 80)
print(f"QUESTION 10: {entry.question}")
print(f"Correct passage ID: {entry.passage_id}")
print("=" * 80)

# Get OpenAI retriever (best - Euclid, 2000)
repository = QdrantOpenAIRepository.get_repository(
    qdrant_client, OPENAI_EMBEDDING_MODEL_NAMES[0], Distance.EUCLID, cache
)
retriever = QdrantRetriever(repository, "clarin-pl-poquad-2000")

# Get retrieval results
print("\nRetrieving with: text-embedding-3-large-Euclid-clarin-pl-poquad-2000")
result = retriever.get_relevant_passages(entry.question)

# Show top 5 passages
passages = [passage for (passage, _) in result.passages]
top_5 = passages[:5]

print(f"\nTOP 5 PASSAGES (n=5):")
print("=" * 80)

for i, passage in enumerate(top_5, 1):
    is_correct = "✓ CORRECT" if passage.id == entry.passage_id else "✗ incorrect"
    print(f"\n{i}. {is_correct}")
    print(f"   ID: {passage.id}")
    print(f"   Text: {passage.context[:200]}...")
    print("-" * 80)

# Check if correct passage is in top 5
correct_in_top5 = entry.passage_id in [p.id for p in top_5]
print(
    f"\n{'✅' if correct_in_top5 else '❌'} Correct passage {'IS' if correct_in_top5 else 'IS NOT'} in top 5"
)
