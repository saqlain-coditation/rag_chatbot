from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.indices.base import BaseIndex
from llama_index.core.llms.llm import LLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor

from ... import lib as rag
from .. import types as t


class NaiveRAGEngine(t.BaseRagSearchEngine):
    def __init__(self, index: BaseIndex, llm: LLM):
        super().__init__()
        self.index = index
        self.llm = llm
        self.engine = self._initialize_engine()

    def _initialize_engine(self) -> BaseQueryEngine:
        retriever = rag.retriever.hybrid_search_retriever(self.index, 10, self.llm)
        reranker = rag.reranker.llm_reranker(self.llm, 3)
        return self._create_engine(retriever, reranker)

    def _create_engine(
        self,
        retriever: BaseRetriever,
        reranker: BaseNodePostprocessor | None = None,
    ) -> BaseQueryEngine:
        return rag.engine.create_query_engine(retriever, self.llm, reranker)

    def search(self, query: str) -> RESPONSE_TYPE:
        return self.engine.query(query)

    async def asearch(self, query) -> RESPONSE_TYPE:
        return await self.engine.aquery(query)
