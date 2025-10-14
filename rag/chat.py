from llama_index.core import Settings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms.llm import LLM

import config
import rag.lib as rag


def setup(
    llm: LLM,
    embedding: BaseEmbedding,
    reader: rag.index.DocumentReader,
):
    Settings.embed_model = embedding
    index = rag.index.load_index(
        reader,
        f"data/store/indexes/{reader.name}",
        rag.index.default_splitter(),
    )
    retriever = rag.retriever.hybrid_search_retriever(index, llm=llm, top_k=10)
    reranker = rag.reranker.llm_reranker(llm, top_n=3)
    engine = rag.engine.create_sub_question_query_engine(
        retriever,
        llm=llm,
        reranker=reranker,
        query_expander=llm,
    )

    return engine


def main():
    reader = rag.index.DocumentReader("data/input/test")
    engine = setup(config.llm, config.embedding, reader)
    choice = ""
    while True:
        choice = input("\nQuestion: ")
        if choice == "x":
            return

        result = engine.query(choice)
        print("Answer: " + str(result), end="\n")
