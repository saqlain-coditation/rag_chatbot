from typing import Dict, List, Optional, Sequence

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.base_selector import MultiSelection, SingleSelection
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.chat_engine.condense_question import CondenseQuestionChatEngine
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.llms.llm import LLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts.base import BasePromptTemplate
from llama_index.core.query_engine import (
    RetrieverQueryEngine,
    RouterQueryEngine,
    SubQuestionQueryEngine,
)
from llama_index.core.question_gen.llm_generators import LLMQuestionGenerator
from llama_index.core.schema import QueryBundle
from llama_index.core.selectors import BaseSelector
from llama_index.core.tools import QueryEngineTool, ToolMetadata


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


def chat_engine_from_query(
    engine: BaseQueryEngine,
    llm: Optional[LLM] = None,
) -> BaseChatEngine:
    return CondenseQuestionChatEngine.from_defaults(engine, llm=llm)


def create_sub_question_query_engine(
    retriever: BaseRetriever,
    llm: Optional[LLM] = None,
    reranker: Optional[BaseNodePostprocessor] = None,
    query_expander: Optional[LLM] = None,
) -> BaseQueryEngine:
    engine = create_query_engine(retriever, llm=llm, reranker=reranker, use_async=True)
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


def router_query_engine(
    engines: List[QueryEngineTool],
    llm: Optional[LLM] = None,
    selector: Optional[BaseSelector] = None,
):
    return RouterQueryEngine.from_defaults(
        query_engine_tools=engines,
        selector=selector or SelectAllSelector(),
        select_multi=True,
        llm=llm,
    )


class SelectAllSelector(BaseSelector):
    def _get_prompts(self) -> Dict[str, BasePromptTemplate]:
        raise NotImplementedError

    def _update_prompts(self, prompts_dict: Dict[str, BasePromptTemplate]) -> None:
        raise NotImplementedError

    def _select(
        self,
        choices: Sequence[ToolMetadata],
        query: QueryBundle,
    ) -> MultiSelection:
        selection = [
            SingleSelection(index=i, reason="All") for i in range(len(choices))
        ]
        return MultiSelection(selections=selection)

    async def _aselect(
        self,
        choices: Sequence[ToolMetadata],
        query: QueryBundle,
    ) -> MultiSelection:
        return self._select(choices, query)
