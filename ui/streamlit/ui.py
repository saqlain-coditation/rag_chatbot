import streamlit as st
import os
import shutil
from ui.indexer import read_index, load_index, create_index, index_path
from ui.engine import aquery_rag
from ui.async_helper import run_async


def save_uploaded_file(uploaded_file, upload_dir):
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(uploaded_file, f)
    return file_path


def main():
    # --- Config ---
    UPLOAD_NAME = "uploads"
    UPLOAD_DIR = f"data/input/{UPLOAD_NAME}"
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    st.set_page_config(page_title="RAG File Manager & Chat", layout="wide")
    st.title("📂 RAG File Manager & 💬 Chat")

    # --- Upload new files ---
    st.header("Add Documents")
    uploaded_files = st.file_uploader(
        "Upload files", type=["pdf", "txt", "docx", "md"], accept_multiple_files=True
    )

    if uploaded_files:
        new_files = 0
        for file in uploaded_files:
            file_path = os.path.join(UPLOAD_DIR, file.name)
            if not os.path.exists(file_path):
                save_uploaded_file(file, UPLOAD_DIR)
                new_files += 1
        if new_files:
            st.success(f"Uploaded {new_files} new file(s)!")
        else:
            st.info("All uploaded files already exist.")

    # --- Always read directory dynamically ---
    st.header("Existing Documents")
    all_files = sorted(os.listdir(UPLOAD_DIR))
    if all_files:
        st.write("Files in directory:")
        for f in all_files:
            st.write(f"- {f}")
    else:
        st.write("No files in the directory yet.")

    # Initialize session state
    if "index" not in st.session_state:
        st.session_state.index = None
        st.session_state.index_path = None

    st.subheader("📦 RAG Index")

    INDEX_DIR = index_path(UPLOAD_NAME)
    if st.session_state.index is None:
        with st.spinner("Loading index..."):
            index = read_index(UPLOAD_NAME)

        if index:
            st.session_state.index = index
            st.session_state.index_path = INDEX_DIR
            st.success("Loaded existing index!")
            st.session_state.index_loaded = True

    # --- If index exists, show rebuild button ---
    if st.session_state.index:
        st.write(f"✅ Index path: `{st.session_state.index_path}`")

        if st.button("🔄 Rebuild Index (Clear & Recreate)"):
            with st.spinner("Rebuilding index from uploaded documents..."):
                import shutil

                shutil.rmtree(INDEX_DIR)
                os.makedirs(INDEX_DIR, exist_ok=True)
                st.session_state.chat_history = []

                index = create_index(UPLOAD_NAME)
                st.session_state.index = index
                st.session_state.index_path = INDEX_DIR
            st.success("Index rebuilt successfully!")
            st.info("💬 Chat history cleared because index was rebuilt.")

    # --- If no index exists, show create button ---
    else:
        st.info("No index found. Please create one from uploaded documents.")
        if st.button("⚙️ Create Index"):
            if "index" not in st.session_state:
                with st.spinner("Creating index from uploaded documents..."):
                    index = load_index(UPLOAD_NAME)
                    st.session_state.index = index
                    st.session_state.index_path = INDEX_DIR
            st.success("Index created successfully!")
            st.rerun()

    # --- Chat interface ---
    st.header("💬 Chat with your RAG Index")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display all previous messages
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["user"])
        with st.chat_message("assistant"):
            st.markdown(chat["bot"])

    # --- Chat input ---
    if prompt := st.chat_input("Ask a question about your documents..."):
        # 1️⃣ Immediately show user message
        st.session_state.chat_history.append({"user": prompt, "bot": None})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2️⃣ Placeholder for bot's response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("_Thinking..._")

            # 3️⃣ Generate RAG answer
            answer = run_async(lambda: aquery_rag(prompt))

            # 4️⃣ Update placeholder with the real answer
            message_placeholder.markdown(answer)

        # 5️⃣ Save conversation
        st.session_state.chat_history[-1]["bot"] = answer
