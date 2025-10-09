from typing import Optional
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.question_gen.llm_generators import LLMQuestionGenerator
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.llms.llm import LLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.chat_engine import CondensePlusContextChatEngine


def create_query_engine(
    retriever: BaseRetriever,
    llm: Optional[LLM] = None,
    reranker: Optional[BaseNodePostprocessor] = None,
    **kwargs,
) -> BaseQueryEngine:
    return RetrieverQueryEngine.from_args(
        retriever,
        llm=llm,
        node_postprocessors=[reranker] if reranker else None,
        **kwargs,
    )


def create_chat_engine(
    retriever: BaseRetriever,
    llm: Optional[LLM] = None,
    reranker: Optional[BaseNodePostprocessor] = None,
    **kwargs,
) -> BaseChatEngine:
    return CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        llm=llm,
        node_postprocessors=[reranker] if reranker else None,
        **kwargs,
    )


def create_sub_question_query_engine(
    retriever: BaseRetriever,
    llm: Optional[LLM] = None,
    reranker: Optional[BaseNodePostprocessor] = None,
    query_expander: Optional[LLM] = None,
) -> BaseQueryEngine:
    engine = create_query_engine(retriever, llm=llm, reranker=reranker)
    return SubQuestionQueryEngine.from_defaults(
        query_engine_tools=[
            QueryEngineTool(
                query_engine=engine,
                metadata=ToolMetadata(
                    name="default_index",
                    description="Active index",
                ),
            ),
        ],
        llm=llm,
        question_gen=LLMQuestionGenerator.from_defaults(llm=query_expander),
    )
