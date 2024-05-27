class Passage:
    def __init__(self, id: str, text: str, title: str):
        self.id = id
        self.text = text
        self.title = title

    def dict(self):
        return {"id": self.id, "text": self.text, "metadata": {"title": self.title}}
