from abc import ABC, abstractmethod
from typing import List

from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.indices.base import BaseIndex
from llama_index.core.schema import Document


class BaseDocumentReader(ABC):
    @abstractmethod
    def read(self) -> List[Document]:
        pass


class BaseIndexManager(ABC):
    def __init__(self):
        super().__init__()
        self._index = None

    def load_index(self) -> BaseIndex:
        if not self._index:
            self._index = self._load_index()
        return self._index

    def add_document(self, document: Document):
        index = self.load_index()
        index.insert(document)

    def add_documents(self, documents: List[Document]):
        for doc in documents:
            self.add_document(doc)

    @abstractmethod
    def create_index(self, documents: List[Document]) -> BaseIndex:
        pass

    @abstractmethod
    def _load_index(self) -> BaseIndex:
        pass


class BaseRagSearchEngine(ABC):
    @abstractmethod
    def search(self, query: str) -> RESPONSE_TYPE:
        pass

    @abstractmethod
    async def asearch(self, query: str) -> RESPONSE_TYPE:
        pass
