import uuid

class EntityId:
    @staticmethod
    def generate():
        return str(uuid.uuid4())