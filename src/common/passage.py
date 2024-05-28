class Passage:
    def __init__(self, id: str, text: str, title: str, start_index: int):
        self.id = id
        self.text = text
        self.title = title
        self.start_index = start_index

    def dict(self):
        return {
            "id": self.id,
            "text": self.text,
            "metadata": {"title": self.title, "start_index": self.start_index},
        }

    def __str__(self) -> str:
        return f"Passage(id={self.id}, text={self.text}, title={self.title}, start_index={self.start_index})"
