from pathlib import Path

import rag.lib as rag

FILE = "data/pump.pdf"
# FILE = "data/Asset-Reference-Plan.pdf"
# 6, 16, 21

parser = rag.readers.parsers.PDFParser()
data = parser.load_data(Path(FILE))
print("\n".join(page.text for page in data))
