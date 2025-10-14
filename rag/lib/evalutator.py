from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.evaluation import (
    AnswerRelevancyEvaluator,
    BatchEvalRunner,
    ContextRelevancyEvaluator,
    CorrectnessEvaluator,
    EvaluationResult,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    RetrievalEvalResult,
    RetrieverEvaluator,
    SemanticSimilarityEvaluator,
    generate_qa_embedding_pairs,
)
from llama_index.core.llama_dataset import LabelledRagDataset
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.llama_pack import download_llama_pack
from llama_index.core.llms.llm import LLM
from llama_index.core.schema import Document

from rag.lib.reader import DocumentReader


def test(
    engine: BaseQueryEngine,
    reader: DocumentReader,
    evaluator: LLM,
):
    dataset = load_dataset(reader, evaluator)
    print(f"Running {len(dataset.examples)} queries...\n")
    for i, example in enumerate(dataset.examples, 1):
        query = example.query
        answer = getattr(example, "reference_answer", None)
        print(f"--- Query {i} ---")
        print(f"Q: {query}\n")

        response = engine.query(query)
        print(f"A: {response.response.strip()}\n")
        print(f"E: {answer.strip()}\n")


async def evaluate(
    engine: BaseQueryEngine,
    reader: DocumentReader,
    evaluator: LLM,
):
    dataset = load_dataset(reader, evaluator)
    benchmark = await evaluate_rag(engine, dataset, llm=evaluator)
    plot_benchmark_df(benchmark)


def load_dataset(reader: DocumentReader, evaluator: LLM) -> LabelledRagDataset:
    try:
        dataset = LabelledRagDataset.from_json(
            f"data/store/datasets/{reader.name}.json"
        )
    except:
        dataset = generate_rag_dataset(
            reader.read_documents(),
            llm=evaluator,
            questions_per_chunk=1,
        )
        dataset.save_json(f"data/store/datasets/{reader.name}.json")

    return dataset


def generate_rag_dataset(
    documents: List[Document],
    llm: LLM,
    questions_per_chunk: int = 10,
) -> LabelledRagDataset:
    dataset_generator = RagDatasetGenerator.from_documents(
        documents=documents,
        llm=llm,
        num_questions_per_chunk=questions_per_chunk,
        show_progress=True,
    )

    dataset = dataset_generator.generate_dataset_from_nodes()
    return dataset


async def evaluate_rag(
    query_engine: BaseQueryEngine,
    rag_dataset: LabelledRagDataset,
    llm: Optional[LLM] = None,
):
    RagEvaluatorPack = download_llama_pack("RagEvaluatorPack", "./pack")
    rag_evaluator = RagEvaluatorPack(
        query_engine=query_engine,
        rag_dataset=rag_dataset,
        judge_llm=llm,
        show_progress=True,
    )
    benchmark_df = await rag_evaluator.arun()
    return benchmark_df


async def evaluate_retriever(
    retriever: BaseRetriever,
    llm: LLM,
    documents: List[Document],
    questions_per_chunk: int = 2,
) -> List[RetrievalEvalResult]:
    dataset = generate_qa_embedding_pairs(
        documents,
        llm=llm,
        show_progress=True,
        num_questions_per_chunk=questions_per_chunk,
    )
    retriever_evaluator = RetrieverEvaluator.from_metric_names(
        ["hit_rate", "mrr", "precision", "recall"], retriever=retriever
    )

    eval_results = await retriever_evaluator.aevaluate_dataset(
        dataset, show_progress=True
    )
    return eval_results


async def evaluate_response(
    engine: BaseQueryEngine,
    llm: LLM,
    rag_dataset: LabelledRagDataset,
) -> Dict[str, List[EvaluationResult]]:
    runner = BatchEvalRunner(
        {
            "faithfulness": FaithfulnessEvaluator(llm=llm),
            "correctness": CorrectnessEvaluator(llm=llm),
            "relevancy": RelevancyEvaluator(llm=llm),
            "answer_relevancy": AnswerRelevancyEvaluator(llm=llm),
            "semantic_similarity": SemanticSimilarityEvaluator(llm=llm),
            "context_relevancy": ContextRelevancyEvaluator(llm=llm),
        },
        workers=8,
        show_progress=True,
    )
    questions = [e.query for e in rag_dataset.examples]
    eval_results = await runner.aevaluate_queries(engine, queries=questions)
    return eval_results


def plot_benchmark_df(benchmark_df):
    # Summary metrics
    summary = benchmark_df.mean(numeric_only=True)
    print(summary)

    # Visualization: bar chart for metrics
    summary.plot(kind="bar", figsize=(8, 4), title="RAG Evaluation Metrics", rot=45)
    plt.ylabel("Score")
    plt.show()


def visualize_retriever_results(eval_results: List[RetrievalEvalResult]):
    df = pd.DataFrame([r.dict() for r in eval_results])

    # Plot average metrics
    avg_metrics = df[["hit_rate", "mrr", "precision", "recall"]].mean()
    avg_metrics.plot(
        kind="bar", figsize=(6, 4), title="Retriever Evaluation Metrics", rot=0
    )
    plt.ylabel("Score")
    plt.show()

    # Optional: visualize metric trends per query
    df[["hit_rate", "recall"]].plot(
        kind="line", figsize=(10, 4), title="Per-query Performance"
    )
    plt.show()

    return df


def visualize_response_results(eval_results: Dict[str, List[EvaluationResult]]):
    # Convert evaluation results dict -> DataFrame
    metrics = {}
    for name, results in eval_results.items():
        metrics[name] = [r.score for r in results if hasattr(r, "score")]

    df = pd.DataFrame(metrics)
    avg = df.mean()

    # Plot average scores
    avg.plot(kind="bar", figsize=(8, 4), title="Response Evaluation Metrics", rot=45)
    plt.ylabel("Average Score")
    plt.show()

    # Optional: per-query trend
    df.plot(kind="line", figsize=(10, 5), title="Response Metric Trends per Query")
    plt.xlabel("Query Index")
    plt.ylabel("Score")
    plt.show()

    return df
