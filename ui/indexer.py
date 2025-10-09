import rag.lib.index as indexer
from rag.lib.reader import DocumentReader


def index_path(upload_name: str = "uploads"):
    return f"data/store/indexes/{upload_name}"


def build_reader(upload_dir: str = "uploads"):
    return DocumentReader(f"data/input/{upload_dir}")


def read_index(upload_dir: str = "uploads"):
    reader = build_reader(upload_dir)
    index = indexer.read_index(index_path(reader.name))
    return index


def load_index(upload_dir: str = "uploads"):
    reader = build_reader(upload_dir)
    index = indexer.load_index(
        reader,
        index_path(reader.name),
        indexer.default_splitter(),
    )
    return index


def create_index(upload_dir: str = "uploads"):
    reader = build_reader(upload_dir)
    index = indexer.create_index(
        reader,
        index_path(reader.name),
        indexer.default_splitter(),
    )
    return index
