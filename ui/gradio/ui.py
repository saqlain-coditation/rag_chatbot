import gradio as gr
import os
from pathlib import Path

from ui.engine import query_rag

# Folder to store uploaded files
UPLOAD_DIR = "data/input/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def upload_files(files):
    """Save uploaded files to the folder"""
    saved_files = []
    for file in files:
        file_path = Path(UPLOAD_DIR) / file.name
        with open(file_path, "wb") as f:
            f.write(file.read())
        saved_files.append(str(file_path))
    return f"Uploaded {len(saved_files)} files."


def ask_question(question):
    """Run RAG on uploaded documents and return the answer"""
    # Assuming your RAG function takes a question and folder path
    answer = query_rag(question)
    return answer


with gr.Blocks() as demo:
    gr.Markdown("## RAG Document QA")

    with gr.Tab("Upload Documents"):
        file_upload = gr.File(
            file_types=[".pdf", ".txt", ".docx"],
            file_count="multiple",
        )
        upload_btn = gr.Button("Upload")
        upload_output = gr.Textbox(label="Upload Status")
        upload_btn.click(upload_files, inputs=file_upload, outputs=upload_output)

    with gr.Tab("Ask Question"):
        question_input = gr.Textbox(label="Your Question")
        answer_output = gr.Textbox(label="Answer")
        ask_btn = gr.Button("Ask")
        ask_btn.click(ask_question, inputs=question_input, outputs=answer_output)


def main():
    demo.launch()
