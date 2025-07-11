from qdrant_client.models import Distance

DATASET_NAMES = ["ipipan/polqa", "clarin-pl/poquad"]

MODEL_NAMES = [
    "sdadas/mmlw-retrieval-roberta-large",
    "ipipan/silver-retriever-base-v1",
    "intfloat/multilingual-e5-large",
    "sdadas/mmlw-roberta-large",
    "BAAI/bge-m3",
]

QA_MODEL_NAMES = ["radlab/polish-qa-v2", "timpal0l/mdeberta-v3-base-squad2"]

INST_MODEL_PATHS = [
    "../../models/Bielik-11B-v2.2-Instruct-q4",
    "../../models/Mistral-7B-Instruct-v0.2-q4",
    "../../models/PLLuM-12-B-instruct-q4",
    "../../models/PLLuM-8B-instruct-q4",
]

OPENAI_EMBEDDING_MODEL_NAMES = ["text-embedding-3-large"]

QUERY_PREFIX_MAP = {
    "sdadas/mmlw-retrieval-roberta-large": "zapytanie: ",
    "ipipan/silver-retriever-base-v1": "Pytanie: ",
    "intfloat/multilingual-e5-large": "query: ",
    "sdadas/mmlw-roberta-large": "zapytanie: ",
    "BAAI/bge-m3": "",
}

PASSAGE_PREFIX_MAP = {
    "sdadas/mmlw-retrieval-roberta-large": "",
    "ipipan/silver-retriever-base-v1": "",
    "intfloat/multilingual-e5-large": "passage: ",
    "sdadas/mmlw-roberta-large": "",
    "BAAI/bge-m3": "",
}

DISTANCES = [Distance.COSINE, Distance.EUCLID]

INDEX_NAMES = [
    "basic_index",
    "polish_index",
    "polish_whitespace_index",
    "polish_stopwords_index",
    "morfologik_index",
    "morfologik_whitespace_index",
    "morfologik_stopwords_index",
]

RERANKER_MODEL_NAMES = [
    "sdadas/polish-reranker-large-ranknet",
]

CHUNK_SIZES = [(500, 100), (1000, 200), (2000, 500), (100000, 0)]

NER_MODEL = "Babelscape/wikineural-multilingual-ner"

HALLUCINATION_MODEL = "vectara/hallucination_evaluation_model"

RERANKER_MODEL = "sdadas/polish-reranker-large-ranknet"

DATASET_SEED = "1234567890"
