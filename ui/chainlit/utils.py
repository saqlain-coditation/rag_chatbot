import os
from pathlib import Path
from typing import List

from chainlit.types import AskFileResponse
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.indices.base import BaseIndex
from llama_index.core.schema import Document
from llama_index.core.workflow import Context

import config
from rag.agent.run import build_agent
from rag.lib.index import default_splitter

Settings.embed_model = config.embedding


def create_index(documents: List[Document] = []):
    index = VectorStoreIndex.from_documents(
        documents,
        transformations=[default_splitter()],
        show_progress=True,
    )
    return index


def import_index(store: List[AskFileResponse]):
    import shutil

    folder_path = ".index/"
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    shutil.rmtree(folder_path)
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    for f in store:
        with open(f.path, "rb") as file:
            data = file.read()
        with open(os.path.join(folder_path, f.name), "wb") as out:
            out.write(data)

    storage_context = StorageContext.from_defaults(persist_dir=folder_path)
    return load_index_from_storage(storage_context)


def load_document(index: BaseIndex, files: List[AskFileResponse]):
    documents = SimpleDirectoryReader(input_files=[f.path for f in files]).load_data()
    for doc in documents:
        index.insert(doc)


def export_index(index: BaseIndex):
    folder_path = ".export"
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=folder_path)


def create_engine(index: BaseIndex):
    return build_agent(index)
