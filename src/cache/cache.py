from pymongo import MongoClient

SIX_MONTHS = 60 * 60 * 24 * 30 * 6


class Cache:
    def __init__(self):
        mongo_client = MongoClient("mongodb://localhost:27017/")
        db = mongo_client["polish-nl-qa"]
        self.key_value_collection = db["key_value"]

    def get(self, key):
        maybe_cached_value = self.key_value_collection.find_one({"key": key})

        if maybe_cached_value is None:
            return None

        return maybe_cached_value["value"]

    def set(self, key, value):
        self.key_value_collection.delete_many({"key": key})
        item = {"key": key, "value": value}
        self.key_value_collection.insert_one(item)

    def unset(self, key):
        return self.key_value_collection.delete_one({"key": key})
