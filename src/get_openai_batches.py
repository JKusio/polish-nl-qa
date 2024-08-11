import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from common.names import (
    CHUNK_SIZES,
    DATASET_NAMES,
    OPENAI_EMBEDDING_MODEL_NAMES,
)
from common.passage_factory import PassageFactory
from common.utils import replace_slash_with_dash
from dataset import dataset_getter
from dataset.poquad_dataset_getter import PoquadDatasetGetter
from dataset.polqa_dataset_getter import PolqaDatasetGetter


def main():
    for model_name in OPENAI_EMBEDDING_MODEL_NAMES:
        get_batch_file(model_name)


def get_passage_factory(
    chunk_size: int, chunk_overlap: int, dataset_name: str
) -> PassageFactory:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, strip_whitespace=True
    )

    dataset_getter = (
        dataset_name == "ipipan/polqa" and PolqaDatasetGetter() or PoquadDatasetGetter()
    )
    return PassageFactory(text_splitter, dataset_getter)


def get_batch_file(model_name: str):
    for dataset_name in DATASET_NAMES:
        for chunk_size, chunk_overlap in CHUNK_SIZES:
            passage_factory = get_passage_factory(
                chunk_size, chunk_overlap, dataset_name
            )

            passages = passage_factory.get_passages()
            filename = replace_slash_with_dash(
                f"{model_name}_{dataset_name}_{chunk_size}_{chunk_overlap}.jsonl"
            )

            with open(
                f"output/batches/{filename}",
                "w",
                encoding="utf-8",
            ) as f:
                for passage in passages:
                    f.write(
                        f"{
                        json.dumps(
                            {
                                "custom_id": f"{passage.id}-{passage.start_index}",
                                "method": "POST",
                                "url": "/v1/embeddings",
                                "body": {"model": model_name, "input": passage.context},
                            },
                            ensure_ascii=False,
                        )
                        }\n"
                    )
                    
    
    for dataset_name in DATASET_NAMES:
        dataset_getter = (
            dataset_name == "ipipan/polqa" and PolqaDatasetGetter() or PoquadDatasetGetter()
        )
        
        all_entries = dataset_getter.get_test_dataset()
        unique_entries = []
        for entry in all_entries:
            if entry.id not in [e.id for e in unique_entries]:
                unique_entries.append(entry)
                
        filename = replace_slash_with_dash(
            f"{dataset_name}.jsonl"
        )
        
        with open(
                f"output/batches/{filename}",
                "w",
                encoding="utf-8",
            ) as f:
                for entry in unique_entries:
                    f.write(
                        f"{
                        json.dumps(
                            {
                                "custom_id": f"{entry.id}",
                                "method": "POST",
                                "url": "/v1/embeddings",
                                "body": {"model": model_name, "input": entry.question},
                            },
                            ensure_ascii=False,
                        )
                        }\n"
                    )
        
        

main()
