## Dependencies:
- Poppler
    - brew install poppler
    - winget install poppler
- Tesseract
    - brew install tesseract
    - winget install tesseract-ocr.tesseract

## Packaging:
`pyinstaller app.spec`

#### Execution Instructions:
- Give execution permissions `chmod +x app`
- Provide input files:
    - Create a directory "input", `mkdir input`
    - Put all documents, text, pdf files in "input" directory
- If available, you can provide index files directly as well. 
    - It will skip indexing step and directly use the data available.
    - Put your index files inside a directory ".index"
- run directly `app`

#### Reading Logs:
- During execution, it will create 3 types of logs:
- Image parsing logs:
    - It will be stored in ".logs/images/"
    - Every image which requires ocr, will added to the images directory.
    - It will also create a txt file with translation with the same name as the image.
- Workflow logs:
    - During execution it will out workflow steps to the console.
    - Judge Query: How good was the given query.
    - Improve Query: New query after improvement.
    - Add Context: New Query with added context.
    - Search: RAG response for the given query.
    - Judge Response: How good was the given response and why.
    - New Query: New query using the previous response to try again.
    - Answer: Final Answer
- Complete Logs:
    - It will be stored in ".logs/log.log"
    - It will contain all the steps, data, llm calls etc.