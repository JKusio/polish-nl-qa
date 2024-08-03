from langchain_text_splitters import RecursiveCharacterTextSplitter
from common.names import DATASET_NAMES
from common.passage_factory import PassageFactory
from dataset.poquad_dataset_getter import PoquadDatasetGetter
from dataset.polqa_dataset_getter import PolqaDatasetGetter


def get_passage_factory(
    chunk_size: int, chunk_overlap: int, dataset_name: str
) -> PassageFactory:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, strip_whitespace=True
    )

    dataset_getter = (
        dataset_name == "ipipan/polqa" and PolqaDatasetGetter() or PoquadDatasetGetter()
    )
    return PassageFactory(text_splitter, dataset_getter)


def main():
    factory = get_passage_factory(1000, 100, DATASET_NAMES[1])

    passages = factory.get_passages()
    print(passages[1])


main()
