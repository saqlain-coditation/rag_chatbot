import asyncio

from llama_index.core import Settings
from llama_index.core.indices.base import BaseIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.workflow import Context

import config

from ..rag.engine import (
    create_query_engine,
    create_sub_question_query_engine,
    router_query_engine,
)
from ..rag.indexes import VectorMemoryIndexManager
from ..rag.readers import GenericReader
from ..rag.reranker import llm_reranker
from ..rag.retriever import hybrid_search_retriever
from .flow import AgenticWorkflow


def build_index(input_dir: str, index_dir: str) -> BaseIndex:
    Settings.embed_model = config.embedding
    index_manager = VectorMemoryIndexManager(index_dir)
    index = index_manager.load_index()
    if index_manager.has_data() == False:
        reader = GenericReader(input_dir)
        documents = reader.read()
        index_manager.add_documents(documents)
    index = index_manager.load_index()
    return index


def build_agent(index: BaseIndex):
    llm = config.llm
    retriever = hybrid_search_retriever(index, 10, llm)
    reranker = llm_reranker(llm, 3)
    engine = router_query_engine(
        engines=[
            QueryEngineTool.from_defaults(
                create_query_engine(retriever, llm, reranker, use_async=True),
                name="simple_query_tool",
                description="Tool for answering simple, single-concept factual questions using direct vector search.",
            ),
            QueryEngineTool.from_defaults(
                create_sub_question_query_engine(retriever, llm, reranker, llm),
                name="subquestion_query_tool",
                description="Tool for answering complex, multi-part questions by breaking them into sub-queries.",
            ),
        ],
        llm=llm,
    )
    agent = AgenticWorkflow(llm, engine, timeout=180)
    return agent


def setup():
    return build_agent(build_index("input", ".index"))


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
