class PassageMetadata:
    def __init__(self, document_id: str):
        self.document_id = document_id

    def dict(self):
        return {
            "document_id": self.document_id
        }