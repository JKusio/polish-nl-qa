class Passage:
    def __init__(
        self,
        id: str,
        title: str,
        context: str,
        start_index: int,
        dataset: str,
        dataset_key: str,
        metadata: dict = {},
    ):
        self.id = id
        self.title = title
        self.context = context
        self.start_index = start_index
        self.dataset = dataset
        self.dataset_key = dataset_key
        self.metadata = metadata

    def dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "context": self.context,
            "start_index": self.start_index,
            "dataset": self.dataset,
            "dataset_key": self.dataset_key,
            "metadata": self.metadata,
        }

    def from_dict(data: dict):
        return Passage(
            id=data["id"],
            title=data["title"],
            context=data["context"],
            start_index=data["start_index"],
            dataset=data["dataset"],
            dataset_key=data["dataset_key"],
            metadata=data["metadata"],
        )

    def __str__(self) -> str:
        return f"Passage(id={self.id}, title={self.title} context={self.context}, title={self.title}, start_index={self.start_index}, dataset={self.dataset}, dataset_key={self.dataset_key}, metadata={self.metadata})"

    def __eq__(self, other):
        if isinstance(other, Passage):
            return self.id == other.id and self.dataset_key == other.dataset_key
        return False

    def __hash__(self):
        return hash(self.id)
