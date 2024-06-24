from qdrant_client.models import Distance, VectorParams


DATASET_NAMES = ["ipipan/polqa", "clarin-pl/poquad"]

MODEL_NAMES = [
    "sdadas/mmlw-retrieval-roberta-large",
    "ipipan/silver-retriever-base-v1",
    "intfloat/multilingual-e5-large",
    "sdadas/mmlw-roberta-large",
    "BAAI/bge-m3",
]

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

SEMANTIC_TYPES = ["interquartile", "standard_deviation", "percentile"]

CHUNK_SIZES = [(500, 100), (1000, 200), (2000, 500)]

CHARACTER_SPLITTING_FUNCTION = [
    "character-500",
    "character-1000",
    "character-2000",
]
