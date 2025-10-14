import asyncio
import logging
from typing import Any, Dict, List, Optional, Sequence

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.base_selector import MultiSelection, SingleSelection
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.callbacks.schema import CBEventType, EventPayload
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
from llama_index.core.query_engine.router_query_engine import acombine_responses
from llama_index.core.question_gen.llm_generators import LLMQuestionGenerator
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.schema import QueryBundle
from llama_index.core.selectors import BaseSelector
from llama_index.core.settings import Settings
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.utils import print_text

logger = logging.getLogger(__name__)


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
    return AsyncRouterQueryEngine.from_defaults(
        query_engine_tools=engines,
        selector=selector,
        llm=llm,
    )


class AsyncRouterQueryEngine(RouterQueryEngine):
    @classmethod
    def from_defaults(
        cls,
        query_engine_tools: Sequence[QueryEngineTool],
        llm: Optional[LLM] = None,
        selector: Optional[BaseSelector] = None,
        summarizer: Optional[TreeSummarize] = None,
        **kwargs: Any,
    ) -> "RouterQueryEngine":
        llm = llm or Settings.llm
        selector = selector or SelectAllSelector()
        return cls(
            selector,
            query_engine_tools,
            llm=llm,
            summarizer=summarizer,
            **kwargs,
        )

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            result = await self._selector.aselect(self._metadatas, query_bundle)

            if len(result.inds) > 1:
                tasks = []
                for i, engine_ind in enumerate(result.inds):
                    log_str = (
                        f"Selecting query engine {engine_ind}: {result.reasons[i]}."
                    )
                    logger.info(log_str)
                    if self._verbose:
                        print_text(log_str + "\n", color="pink")
                    selected_query_engine = self._query_engines[engine_ind]
                    tasks.append(selected_query_engine.aquery(query_bundle))

                # Modified to support true async
                responses = await asyncio.gather(*tasks)
                if len(responses) > 1:
                    final_response = await acombine_responses(
                        self._summarizer, responses, query_bundle
                    )
                else:
                    final_response = responses[0]
            else:
                try:
                    selected_query_engine = self._query_engines[result.ind]
                    log_str = f"Selecting query engine {result.ind}: {result.reason}."
                    logger.info(log_str)
                    if self._verbose:
                        print_text(log_str + "\n", color="pink")
                except ValueError as e:
                    raise ValueError("Failed to select query engine") from e

                final_response = await selected_query_engine.aquery(query_bundle)

            # add selected result
            final_response.metadata = final_response.metadata or {}
            final_response.metadata["selector_result"] = result

            query_event.on_end(payload={EventPayload.RESPONSE: final_response})

        return final_response


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
