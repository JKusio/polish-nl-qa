from common.passage_metadata import PassageMetadata


class Passage:
    def __init__(self, id: str, text: str, metadata: PassageMetadata):
        self.id = id
        self.text = text
        self.metadata = metadata

    def dict(self):
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata.dict()
        }