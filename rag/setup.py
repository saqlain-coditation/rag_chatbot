from llama_index.core import Settings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms.llm import LLM

from config import embedding, evaluator, llm
from rag.lib.engine import create_query_engine, create_sub_question_query_engine
from rag.lib.evalutator import evaluate, test
from rag.lib.index import default_splitter, load_index
from rag.lib.reader import DocumentReader
from rag.lib.reranker import llm_reranker
from rag.lib.retriever import hybrid_search_retriever


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
