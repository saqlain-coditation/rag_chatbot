from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.postprocessor import SentenceTransformerRerank, LLMRerank
from llama_index.core.llms.llm import LLM

default_cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-2-v2"
default_colbert_model = "colbert-ir/colbertv2.0"


def llm_reranker(llm: LLM, top_n: int) -> BaseNodePostprocessor:
    return LLMRerank(llm=llm, top_n=top_n)


# pip install sentence-transformers
# Requires torch
def cross_encoder_reranker(model_name: str, top_n: int) -> BaseNodePostprocessor:
    return SentenceTransformerRerank(model=model_name, top_n=top_n)


# pip install llama-index-postprocessor-colbert-rerank
# Requires torch
def colbert_reranker(model_name: str, top_n: int) -> BaseNodePostprocessor:
    from llama_index.postprocessor.colbert_rerank import ColbertRerank

    return ColbertRerank(
        top_n=top_n,
        model=model_name,
        tokenizer=model_name,
        keep_retrieval_score=True,
    )
