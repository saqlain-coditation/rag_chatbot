## Dependencies:
- Poppler
    - brew install poppler
    - winget install poppler
- Tesseract
    - brew install tesseract
    - winget install tesseract-ocr.tesseract

## Packaging:
`pyinstaller --onefile app.py --hidden-import=tiktoken_ext.openai_public --hidden-import=tiktoken_ext`