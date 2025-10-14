from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.indices.base import BaseIndex
from llama_index.core.llms.llm import LLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor

from ... import lib as rag
from .naive_rag_engine import NaiveRAGEngine


class SubQuestionRagEngine(NaiveRAGEngine):
    def _create_engine(
        self,
        retriever: BaseRetriever,
        reranker: BaseNodePostprocessor | None = None,
    ) -> BaseQueryEngine:
        return rag.engine.create_sub_question_query_engine(
            retriever, self.llm, reranker, self.llm
        )
