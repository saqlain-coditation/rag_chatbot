from typing import Optional, Type
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document


class DocumentReader:
    def __init__(self, directory: str, reader_cls: Optional[Type] = None):
        self.directory = directory
        self.reader_cls = reader_cls or SimpleDirectoryReader
        self.documents: Optional[list[Document]] = None
        self.name = directory.split("/")[-1]

    def read_documents(self) -> list[Document]:
        return self.documents or self.load_documents()

    def load_documents(self) -> list[Document]:
        if not getattr(self, "reader", None):
            self.reader = self.reader_cls(self.directory)
        self.documents = self.reader.load_data(show_progress=True)
        return self.documents
