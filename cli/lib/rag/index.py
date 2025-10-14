from typing import Optional

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.indices.base import BaseIndex
from llama_index.core.node_parser import SentenceSplitter, TextSplitter

from lib.rag.reader import DocumentReader


def default_splitter(
    chunk_size: int = 256,
    chunk_overlap: Optional[int] = None,
) -> TextSplitter:
    if not chunk_overlap:
        chunk_overlap = chunk_size // 8
    return SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def load_index(
    reader: DocumentReader,
    storage: str,
    splitter: Optional[TextSplitter] = None,
) -> BaseIndex:
    index = read_index(storage) or create_index(reader, storage, splitter)
    return index


def read_index(storage: str) -> BaseIndex:
    try:
        storage_context = StorageContext.from_defaults(persist_dir=storage)
        return load_index_from_storage(storage_context)
    except Exception as e:
        return None


def create_index(
    reader: DocumentReader,
    storage: str,
    splitter: Optional[TextSplitter] = None,
) -> BaseIndex:
    documents = reader.read_documents()
    transformers = []
    if splitter:
        transformers.append(splitter)
    index = VectorStoreIndex.from_documents(
        documents, transformations=transformers, show_progress=True
    )
    index.storage_context.persist(storage)
    return index


def index_from_engine(engine: BaseQueryEngine) -> BaseIndex:
    return engine._retriever._index


def print_index_data(index: BaseIndex):
    print("Printing all vector data:")
    storage_context = index.storage_context
    vector_store = storage_context.vector_store
    docstore = storage_context.docstore
    for node_id, vector in vector_store.data.embedding_dict.items():
        node = docstore.get_node(node_id)
        print(f"Node ID: {node_id}")
        print(node.text)
        print("-" * 20)
