from typing import Dict

from llama_index.core import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

from .. import types as t
from .parsers import PDFParser


class GenericReader(t.BaseDocumentReader, SimpleDirectoryReader):
    file_extractors: Dict[str, BaseReader] = {
        ".pdf": PDFParser(),
    }

    def __init__(
        self,
        input_dir=None,
        input_files=None,
        exclude=None,
        exclude_hidden=True,
        exclude_empty=False,
        errors="ignore",
        recursive=False,
        encoding="utf-8",
        filename_as_id=False,
        required_exts=None,
        file_extractor=None,
        num_files_limit=None,
        file_metadata=None,
        raise_on_error=False,
        fs=None,
    ):
        file_extractor = (file_extractor or {}) | self.file_extractors
        super().__init__(
            input_dir,
            input_files,
            exclude,
            exclude_hidden,
            exclude_empty,
            errors,
            recursive,
            encoding,
            filename_as_id,
            required_exts,
            file_extractor,
            num_files_limit,
            file_metadata,
            raise_on_error,
            fs,
        )
        self.documents = None

    def read(self):
        self.documents = self.load_data(show_progress=True)
        return self.documents

    def get_documents(self) -> list[Document]:
        return self.documents or self.read()
