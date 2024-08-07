from qdrant_client.models import Distance

DATASET_NAMES = ["ipipan/polqa", "clarin-pl/poquad"]

MODEL_NAMES = [
    "sdadas/mmlw-retrieval-roberta-large",
    "ipipan/silver-retriever-base-v1",
    "intfloat/multilingual-e5-large",
    "sdadas/mmlw-roberta-large",
    "BAAI/bge-m3",
]

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
    "BAAI/bge-reranker-v2-gemma",
    "unicamp-dl/mt5-13b-mmarco-100k",
]

CHUNK_SIZES = [(500, 100), (1000, 200), (2000, 500), (100000, 0)]
