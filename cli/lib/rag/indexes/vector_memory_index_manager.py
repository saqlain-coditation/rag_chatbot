import os
from typing import List

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.indices.base import BaseIndex
from llama_index.core.schema import Document

from ... import rag as rag
from .. import types as t


class VectorMemoryIndexManager(t.BaseIndexManager):
    def __init__(self, persist_dir: str):
        super().__init__()
        self._persist_dir = persist_dir
        self._splitter = rag.index.default_splitter()
        self._documents = None

    def add_document(self, document):
        super().add_document(document)
        self._persist(self._index)
        if self._documents is not None:
            self._documents.append(document)

    def create_index(self, documents: List[Document]) -> BaseIndex:
        import shutil

        shutil.rmtree(self._persist_dir)
        os.makedirs(self._persist_dir, exist_ok=True)
        self._index = self._create_index(documents)
        self._persist(self._index)
        self._documents = documents
        return self._index

    def has_data(self) -> bool | None:
        if self._documents is None:
            return None
        else:
            return len(self._documents) > 0

    def _load_index(self) -> BaseIndex:
        try:
            storage_context = StorageContext.from_defaults(
                persist_dir=self._persist_dir
            )
            return load_index_from_storage(storage_context)
        except Exception as e:
            self._documents = []
            return VectorStoreIndex.from_documents(
                self._documents,
                transformations=[self._splitter],
                show_progress=True,
            )

    def _create_index(self, documents: List[Document]) -> BaseIndex:
        return VectorStoreIndex.from_documents(
            documents,
            transformations=[self._splitter],
            show_progress=True,
        )

    def _persist(self, index: BaseIndex):
        index.storage_context.persist(self._persist_dir)
