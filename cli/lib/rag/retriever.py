from typing import List

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.indices.base import BaseIndex
from llama_index.core.llms.llm import LLM
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.retrievers.bm25 import BM25Retriever


def hybrid_search_retriever(
    index: BaseIndex,
    top_k: int = 2,
    llm: LLM = None,
    queries: int = 4,
) -> BaseRetriever:
    main_retriever = default_retriever(index, top_k)
    bm25_retriever = BM25Retriever.from_defaults(
        docstore=index.docstore, similarity_top_k=top_k
    )
    retriever = AsyncQueryFusionRetriever(
        [main_retriever, bm25_retriever],
        llm=llm,
        similarity_top_k=top_k,
        num_queries=queries,  # set this to 1 to disable query generation
        mode=FUSION_MODES.RECIPROCAL_RANK,
        use_async=True,
    )

    return retriever


def default_retriever(index: BaseIndex, top_k: int = 2) -> BaseRetriever:
    return index.as_retriever(similarity_top_k=top_k)


def vector_retriever(index: BaseIndex, top_k: int = 2, alpha: int = 1) -> BaseRetriever:
    return VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
        alpha=alpha,
        node_ids=list(index.index_struct.nodes_dict.values()),
    )


class AsyncQueryFusionRetriever(QueryFusionRetriever):
    async def _aget_queries(self, original_query: str) -> List[QueryBundle]:
        prompt_str = self.query_gen_prompt.format(
            num_queries=self.num_queries - 1,
            query=original_query,
        )

        # Modified to support true async
        response = await self._llm.acomplete(prompt_str)

        # Strip code block and assume LLM properly put each query on a newline
        queries = response.text.strip("`").split("\n")
        queries = [q.strip() for q in queries if q.strip()]
        if self._verbose:
            queries_str = "\n".join(queries)
            print(f"Generated queries:\n{queries_str}")

        # The LLM often returns more queries than we asked for, so trim the list.
        return [QueryBundle(q) for q in queries[: self.num_queries - 1]]

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        queries: List[QueryBundle] = [query_bundle]
        if self.num_queries > 1:
            # Modified to support true async
            queries.extend(await self._aget_queries(query_bundle.query_str))

        results = await self._run_async_queries(queries)

        if self.mode == FUSION_MODES.RECIPROCAL_RANK:
            return self._reciprocal_rerank_fusion(results)[: self.similarity_top_k]
        elif self.mode == FUSION_MODES.RELATIVE_SCORE:
            return self._relative_score_fusion(results)[: self.similarity_top_k]
        elif self.mode == FUSION_MODES.DIST_BASED_SCORE:
            return self._relative_score_fusion(results, dist_based=True)[
                : self.similarity_top_k
            ]
        elif self.mode == FUSION_MODES.SIMPLE:
            return self._simple_fusion(results)[: self.similarity_top_k]
        else:
            raise ValueError(f"Invalid fusion mode: {self.mode}")
