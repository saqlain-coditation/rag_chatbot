import asyncio

from llama_index.core import Settings
from llama_index.core.indices.base import BaseIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.workflow import Context

import config

from .. import lib as rag
from .flow import AgenticWorkflow


def build_index(input_dir: str, index_dir: str) -> BaseIndex:
    Settings.embed_model = config.embedding
    index_manager = rag.indexes.VectorMemoryIndexManager(index_dir)
    index = index_manager.load_index()
    if index_manager.has_data() == False:
        reader = rag.readers.GenericReader(input_dir)
        documents = reader.read()
        index_manager.add_documents(documents)
    index = index_manager.load_index()
    return index


def build_agent(index: BaseIndex):
    llm = config.llm
    retriever = rag.retriever.hybrid_search_retriever(index, 10, llm)
    reranker = rag.reranker.llm_reranker(llm, 3)
    engine = rag.engine.router_query_engine(
        engines=[
            QueryEngineTool.from_defaults(
                rag.engine.create_query_engine(
                    retriever, llm, reranker, use_async=True
                ),
                name="simple_query_tool",
                description="Tool for answering simple, single-concept factual questions using direct vector search.",
            ),
            QueryEngineTool.from_defaults(
                rag.engine.create_sub_question_query_engine(
                    retriever, llm, reranker, llm
                ),
                name="subquestion_query_tool",
                description="Tool for answering complex, multi-part questions by breaking them into sub-queries.",
            ),
        ],
        llm=llm,
    )
    agent = AgenticWorkflow(llm, engine, timeout=180)
    return agent


def setup():
    return build_agent(build_index("data/store/indexes/test", "data/input/test"))


async def _main():
    agent = setup()
    ctx = Context(agent)
    choice = ""
    while True:
        choice = input("\nQuestion: ")
        if choice == "x":
            return

        result = await agent.run(query=choice, ctx=ctx)
        print("Answer: " + str(result), end="\n")


def main():
    asyncio.run(_main())
