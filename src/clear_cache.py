from pymongo import MongoClient


client = MongoClient("mongodb://localhost:27017/")
db = client["polish-nl-qa"]
collection_name = "key_value"
# prefixes = ["count", "vectorizer", "prompt", "reranker", "query", "score"]
prefixes = []
collection = db[collection_name]

for prefix in prefixes:
    print(f"Clearing keys with prefix: {prefix}")
    collection.delete_many({"key": {"$regex": f"^{prefix}:.*"}})

client.close()
