import streamlit as st
from rag.lib.retriever import hybrid_search_retriever
from rag.lib.reranker import llm_reranker
from rag.lib.engine import create_chat_engine

from config import llm
from ui.indexer import load_index

engine = None


def build_engine():
    global engine
    if engine:
        return engine

    retriever = hybrid_search_retriever(load_index(), llm=llm, top_k=2)
    reranker = llm_reranker(llm, top_n=2)
    engine = create_chat_engine(
        retriever,
        llm=llm,
        reranker=reranker,
        query_expander=llm,
    )
    return engine


def query_rag(query: str):
    if "engine" not in st.session_state:
        st.session_state.engine = build_engine()
    engine = st.session_state.engine
    response = engine.chat(query)
    return str(response)


async def aquery_rag(query: str):
    engine = build_engine()
    response = await engine.achat(query)
    return str(response)
