import redis

SIX_MONTHS = 60 * 60 * 24 * 30 * 6


class Cache:
    def __init__(self):
        self.redis = redis.Redis(host="localhost", port=6379, db=0)

    def get(self, key):
        return self.redis.get(key)

    def set(self, key, value):
        return self.redis.set(key, value, SIX_MONTHS)
