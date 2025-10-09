from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.indices.base import BaseIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.llms.llm import LLM


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
    retriever = QueryFusionRetriever(
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
