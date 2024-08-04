import redis

r = redis.StrictRedis(host="localhost", port=6379, db=0)

# comment out the keys you want to keep
prefixes = [
    "count:*",
    "vectorizer:*",
    "prompt:*",
    "reranker:*",
    "query:*",
]

for prefix in prefixes:
    print(f"Clearing keys with prefix: {prefix}")
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor, match=prefix)
        print(f"Deleting {len(keys)} keys")
        if len(keys) > 0:
            r.delete(*keys)
        elif cursor == 0:
            break
        else:
            print("No keys found")
            break
