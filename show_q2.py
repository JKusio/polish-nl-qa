#!/usr/bin/env python3
import sys, os

sys.path.insert(0, "src")

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from cache.cache import Cache
from repository.qdrant_repository import QdrantRepository
from retrievers.qdrant_retriever import QdrantRetriever
from vectorizer.hf_vectorizer import HFVectorizer
from common.utils import get_dataset_key
import json

# Load question
with open("output/curated_datasets/curated_poquad_100.json", "r") as f:
    questions = json.load(f)

question_data = questions[19]  # Pytanie #20 (index 19)
question = question_data["question"]
answers = question_data["answers"]
correct_id = question_data.get("passage_id")

print(f"\n{'=' * 100}")
print(f"PYTANIE #20: {question}")
print(f"ODPOWIEDŹ: {', '.join(answers)}")
print("=" * 100)

# Setup retriever
client = QdrantClient(host="localhost", port=6333)
cache = Cache()

model_name = "sdadas/mmlw-retrieval-roberta-large"
distance = Distance.EUCLID
collection_name = "sdadas-mmlw-retrieval-roberta-large-Euclid"
dataset_key = get_dataset_key("clarin-pl/poquad", 1000)

from common.models_dimensions import MODEL_DIMENSIONS_MAP

dimension = MODEL_DIMENSIONS_MAP[model_name]
vectors_config = VectorParams(size=dimension, distance=distance)

vectorizer = HFVectorizer(model_name, cache)

repository = QdrantRepository(
    client,
    collection_name,
    model_name,
    vectors_config,
    vectorizer,
    cache,
    "",
    "zapytanie: ",
)

retriever = QdrantRetriever(repository, dataset_key)

# Get results
print("\nSzukanie pasaży (TOP 5)...\n")
result = retriever.get_relevant_passages(question, size=5)

# Show passages
for i, (passage, score) in enumerate(result.passages, 1):
    is_correct = "✓ POPRAWNY" if correct_id and passage.id == correct_id else ""
    print(f"{'#' * 100}")
    print(f"PASAŻ #{i} (Score: {score:.4f}) {is_correct}")
    print("#" * 100)
    print(f"Tytuł: {passage.title}")
    print(f"\n{passage.context}\n")
    print(f"ID: {passage.id}")
    print()

# Summary
found_at = None
for i, (passage, _) in enumerate(result.passages, 1):
    if passage.id == correct_id:
        found_at = i
        break

print("=" * 100)
if found_at:
    print(f"✓✓✓ POPRAWNY PASAŻ NA POZYCJI {found_at} ✓✓✓")
else:
    print(f"✗✗✗ POPRAWNY PASAŻ NIE W TOP-5 ✗✗✗")
print("=" * 100)
