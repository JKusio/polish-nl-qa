import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from cache.cache import Cache
from common.names import (
    CHUNK_SIZES,
    DATASET_NAMES,
    DATASET_SEED,
    DISTANCES,
    OPENAI_EMBEDDING_MODEL_NAMES,
)
from common.passage_factory import PassageFactory
from common.utils import get_all_openai_model_combinations, get_generator_hash, replace_slash_with_dash
from dataset import dataset_getter
from dataset.poquad_dataset_getter import PoquadDatasetGetter
from dataset.polqa_dataset_getter import PolqaDatasetGetter
from repository.qdrant_openai_repository import QdrantOpenAIRepository
from qdrant_client.models import VectorParams, PointStruct, Distance

from retrievers.qdrant_retriever import QdrantRetriever
from retrievers.retriever import Retriever




def main():
    qdrant_client = QdrantClient(host="localhost", port=6333)
    cache = Cache()
    
    get_batch_file(qdrant_client, cache)


def get_retriever(
    qdrantClient: QdrantClient, distance: Distance, cache: Cache, dataset_key: str
) -> Retriever:
    repository = QdrantOpenAIRepository.get_repository(
        qdrantClient, OPENAI_EMBEDDING_MODEL_NAMES[0], distance, cache
    )
    
    retriever = QdrantRetriever(repository, dataset_key)

    return retriever


def get_batch_file(qdrantClient: QdrantClient, cache: Cache):
    combinations = get_all_openai_model_combinations()
    
    poquad_dataset_getter = PoquadDatasetGetter()
    polqa_dataset_getter = PolqaDatasetGetter()

    poquad_dataset = poquad_dataset_getter.get_random_n_test(500, DATASET_SEED)[:100]
    polqa_dataset = polqa_dataset_getter.get_random_n_test(500, DATASET_SEED)[:100]
    
    custom_ids = set()
    
    for model, distance, dataset_key in combinations:
        retriever = get_retriever(
            qdrantClient, 
            distance,
            cache,
            dataset_key
        )
        
        dataset = poquad_dataset if "poquad" in dataset_key else polqa_dataset
        for entry in dataset:
            filename = replace_slash_with_dash(
                f"{"gpt-4o-mini"}_{dataset_key}.jsonl"
            )
            file_path = f"output/batches/{filename}"
            
            result = retriever.get_relevant_passages(entry.question)
            
            ns = [1, 3, 5, 10]
            
            for n in ns:
                passages = [passage for (passage, _) in result.passages]
                top_n_passages = passages[:n]
                
                context = " ".join([passage.context for passage in top_n_passages]).replace("\n", " ")

                hash_key = get_generator_hash(entry.question, context, "instruction", "gpt-4o-mini")
                
                if (hash_key in custom_ids):
                    continue
                
                custom_ids.add(hash_key)
                
                prompt = f"""
                [INST]
                    Wygeneruj krótką odpowiedź na pytanie wyłącznie na podstawie poniższego kontekstu:
                    {context}

                    Pytanie: {entry.question}
                [/INST]
                """
                
                if os.path.exists(file_path):
                    # Open the file in append mode
                    with open(file_path, "a", encoding="utf-8") as f:
                        f.write(
                            f"{
                            json.dumps(
                                {
                                    "custom_id": f"{hash_key}",
                                    "method": "POST",
                                    "url": "/v1/chat/completions",
                                    "body": {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt.replace("\n", " ")}],"max_tokens": 1000},
                                },
                                ensure_ascii=False,
                            )
                            }\n"
                        )
                else:
                    # Open the file in write mode
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(
                            f"{
                            json.dumps(
                                {
                                    "custom_id": f"{hash_key}",
                                    "method": "POST",
                                    "url": "/v1/chat/completions",
                                    "body": {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt.replace("\n", " ")}],"max_tokens": 1000},
                                },
                                ensure_ascii=False,
                            )
                            }\n"
                        )

main()
