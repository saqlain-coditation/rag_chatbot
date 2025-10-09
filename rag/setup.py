from llama_index.core import Settings

from llama_index.core.llms.llm import LLM
from llama_index.core.base.embeddings.base import BaseEmbedding

from config import llm, embedding, evaluator

from rag.lib.reader import DocumentReader
from rag.lib.index import load_index, default_splitter
from rag.lib.retriever import hybrid_search_retriever
from rag.lib.reranker import llm_reranker
from rag.lib.engine import create_sub_question_query_engine, create_query_engine
from rag.lib.evalutator import evaluate, test


def setup(
    llm: LLM,
    embedding: BaseEmbedding,
    reader: DocumentReader,
):
    Settings.embed_model = embedding
    index = load_index(
        reader,
        f"data/store/indexes/{reader.name}",
        default_splitter(),
    )
    retriever = hybrid_search_retriever(index, llm=llm, top_k=2)
    reranker = llm_reranker(llm, top_n=2)
    engine = create_query_engine(
        retriever,
        llm=llm,
        reranker=reranker,
        query_expander=llm,
    )

    return engine


def main():
    reader = DocumentReader("data/input/essay")
    engine = setup(llm, embedding, reader)
    test(engine, reader, evaluator)
