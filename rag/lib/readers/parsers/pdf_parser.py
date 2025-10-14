import io
import os
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, List, Optional, Union

from fsspec import AbstractFileSystem
from llama_index.core.readers.base import BaseReader
from llama_index.core.readers.file.base import get_default_fs, is_default_fs
from llama_index.core.schema import Document
from PIL import Image
from PIL.ImageFile import ImageFile
from regex import T
from tenacity import retry, stop_after_attempt

from rag.lib.utils.ocr import ocr

RETRY_TIMES = 3


class PDFParser(BaseReader):

    def __init__(
        self,
        return_full_document: Optional[bool] = False,
    ) -> None:
        """
        Initialize PDFReader.
        """
        self.return_full_document = return_full_document

    @retry(stop=stop_after_attempt(RETRY_TIMES))
    def load_data(
        self,
        file: Union[Path, PurePosixPath],
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Parse file."""
        fs = fs or get_default_fs()
        _Path = Path if is_default_fs(fs) else PurePosixPath
        if not isinstance(file, (Path, PurePosixPath)):
            file = _Path(file)

        try:
            import pdfplumber
        except ImportError:
            raise ImportError(
                "pdfplumber is required to read PDF files: `pip install pdfplumber`"
            )

        try:
            import pymupdf
        except ImportError:
            raise ImportError(
                "pymupdf is required to read PDF files: `pip install pymupdf`"
            )

        with fs.open(str(file), "rb") as fp:
            stream = io.BytesIO(fp.read())

            pages: List[str] = []
            with pdfplumber.open(stream) as pdf, pymupdf.open(
                stream=stream, filetype="pdf"
            ) as pdf_meta:
                for i, page in enumerate(pdf.pages):
                    meta_page = pdf_meta[min(i, len(pdf_meta))]
                    text = page.extract_text()

                    page_ocr = ""
                    for image in self._extract_images(meta_page, pdf_meta):
                        ocr = self._parse_image(image)
                        page_ocr += ocr

                    if not page_ocr and not text:
                        # Either the page was empty or was unparseable
                        # Try ocr again, with the entire page as image
                        pix = meta_page.get_pixmap()  # default resolution
                        image = Image.open(io.BytesIO(pix.tobytes("jpeg")))
                        ocr = self._parse_image(image)
                        page_ocr += ocr

                    text += page_ocr
                    pages.append(text)

            docs = []
            # This block returns a whole PDF as a single Document
            if self.return_full_document:
                metadata = {"file_name": file.name}
                if extra_info is not None:
                    metadata.update(extra_info)

                # Join text extracted from each page
                text = "\n".join(pages)
                docs.append(Document(text=text, metadata=metadata))

            # This block returns each page of a PDF as its own Document
            else:
                # Iterate over every page

                for i, page in enumerate(pages):
                    page_label = str(i + 1)
                    metadata = {"page_label": page_label, "file_name": file.name}
                    if extra_info is not None:
                        metadata.update(extra_info)

                    docs.append(Document(text=page, metadata=metadata))

            return docs

    def _extract_images(self, page, document) -> List[ImageFile]:
        images = []
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = document.extract_image(xref)
            image_bytes = base_image["image"]
            pil_image = Image.open(io.BytesIO(image_bytes))
            images.append(pil_image)
        return images

    def _parse_image(self, image: ImageFile) -> str:
        text = PDFParser._parse_image_local(image)
        if len(text) < 10:
            return text
        else:
            return ocr(image) or ""

    @staticmethod
    def _parse_image_local(image: ImageFile) -> str:
        try:
            import pytesseract
        except ImportError:
            raise ImportError(
                "pytesseract is required to read PDF files: `pip install pytesseract`"
                "Also install tesseract: `https://tesseract-ocr.github.io/tessdoc/Installation.html`"
            )
        ocr = pytesseract.image_to_string(image, lang="eng")
        return ocr.strip()
