#!/usr/bin/env python3
# examples/run_evaluation_example.py
"""
Example script demonstrating how to use the evaluation system.
"""

import asyncio
import logging
from pathlib import Path

from src.utils.logging_config import setup_logging
from src.evaluation.dataset import EvaluationDataset, EvaluationSample
from src.evaluation.evaluator import Evaluator
from src.evaluation.reporter import EvaluationReporter
from src.evaluation.metrics import (
    calculate_accuracy_metrics,
    calculate_relevance_metrics,
    calculate_faithfulness,
)
from src.graph.workflow import enable_instrumentation


# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def example_metrics_calculation():
    """Example: Calculate metrics directly without running the workflow."""
    print("\n" + "=" * 60)
    print("Example 1: Direct Metrics Calculation")
    print("=" * 60)
    
    # Sample data
    predicted = "SQL injection attacks exploit vulnerabilities in web applications by injecting malicious SQL code."
    ground_truth = "SQL injection is a code injection technique that exploits security vulnerabilities in an application's database layer."
    question = "What is SQL injection?"
    contexts = [
        "SQL injection (SQLi) is a web security vulnerability that allows an attacker to interfere with queries.",
        "Common prevention methods include parameterized queries and input validation.",
    ]
    
    # Calculate accuracy metrics
    accuracy = calculate_accuracy_metrics(predicted, ground_truth)
    print("\nAccuracy Metrics:")
    print(f"  Semantic Similarity: {accuracy['semantic_similarity']:.4f}")
    print(f"  ROUGE-1: {accuracy['rouge_scores']['rouge1']:.4f}")
    print(f"  ROUGE-L: {accuracy['rouge_scores']['rougeL']:.4f}")
    print(f"  BLEU: {accuracy['bleu_score']:.4f}")
    
    # Calculate relevance metrics
    relevance = calculate_relevance_metrics(question, predicted, contexts)
    print("\nRelevance Metrics:")
    print(f"  Answer Relevance: {relevance['answer_relevance']:.4f}")
    print(f"  Context Relevance (mean): {relevance['context_relevance']['mean']:.4f}")
    
    # Calculate faithfulness
    faithfulness = calculate_faithfulness(predicted, contexts)
    print("\nFaithfulness Metrics:")
    print(f"  Faithfulness Score: {faithfulness['faithfulness_score']:.4f}")
    print(f"  Hallucination Score: {faithfulness['hallucination_detection']['hallucination_score']:.4f}")


def example_dataset_operations():
    """Example: Working with evaluation datasets."""
    print("\n" + "=" * 60)
    print("Example 2: Dataset Operations")
    print("=" * 60)
    
    # Create a dataset programmatically
    dataset = EvaluationDataset()
    
    # Add samples
    dataset.add_sample(EvaluationSample(
        id="example_001",
        question="What is a brute force attack?",
        ground_truth_answer="A brute force attack systematically tries all possible passwords.",
        question_type="cybersecurity_knowledge",
        difficulty="easy",
    ))
    
    dataset.add_sample(EvaluationSample(
        id="example_002",
        question="Show authentication failures from the logs",
        ground_truth_answer="The logs show multiple authentication failures for various users.",
        question_type="log_analysis",
        difficulty="medium",
    ))
    
    print(f"\nDataset created with {len(dataset)} samples")
    print(f"Summary: {dataset.summary()}")
    
    # Filter samples
    log_samples = dataset.get_samples_by_type("log_analysis")
    print(f"Log analysis samples: {len(log_samples)}")
    
    easy_samples = dataset.get_samples_by_difficulty("easy")
    print(f"Easy samples: {len(easy_samples)}")
    
    # Save to file
    output_path = Path("/tmp/example_dataset.json")
    dataset.to_json(output_path)
    print(f"\nDataset saved to: {output_path}")
    
    # Load from file
    loaded_dataset = EvaluationDataset.from_json(output_path)
    print(f"Loaded dataset with {len(loaded_dataset)} samples")


def example_load_sample_dataset():
    """Example: Load and inspect the sample evaluation dataset."""
    print("\n" + "=" * 60)
    print("Example 3: Load Sample Dataset")
    print("=" * 60)
    
    dataset_path = Path("data/evaluation/sample_eval_dataset.json")
    
    if not dataset_path.exists():
        print(f"Sample dataset not found at {dataset_path}")
        return None
    
    dataset = EvaluationDataset.from_json(dataset_path)
    
    print(f"\nLoaded {len(dataset)} samples")
    print(f"Summary: {dataset.summary()}")
    
    print("\nSample questions:")
    for i, sample in enumerate(dataset[:3]):
        print(f"  {i+1}. [{sample.question_type}] {sample.question[:60]}...")
    
    return dataset


async def example_run_evaluation(dataset: EvaluationDataset):
    """Example: Run evaluation on a dataset (requires configured environment)."""
    print("\n" + "=" * 60)
    print("Example 4: Run Evaluation")
    print("=" * 60)
    
    print("\nNOTE: This example requires a configured environment with:")
    print("  - Neo4j database connection")
    print("  - MCP RDF server")
    print("  - API keys for LLM")
    print("\nSkipping actual execution. See the code for usage patterns.")
    
    # The code below shows how to run the evaluation
    # Uncomment when environment is configured
    
    """
    # Enable instrumentation for timing metrics
    enable_instrumentation(True)
    
    # Create evaluator
    evaluator = Evaluator(config={
        "evaluation": {"batch_size": 1, "max_iterations": 3},
    })
    
    # Run evaluation (using only first 2 samples for demo)
    subset = EvaluationDataset(samples=list(dataset)[:2])
    results = await evaluator.evaluate_dataset(subset)
    
    # Generate summary
    summary = evaluator.generate_summary(subset)
    
    print(f"\nEvaluation Summary:")
    print(f"  Total Samples: {summary.total_samples}")
    print(f"  Successful: {summary.successful_evaluations}")
    print(f"  Failed: {summary.failed_evaluations}")
    print(f"  Mean Execution Time: {summary.aggregate_speed.get('mean_total_time', 0):.2f}s")
    print(f"  Mean Semantic Similarity: {summary.aggregate_accuracy.get('mean_semantic_similarity', 0):.4f}")
    
    # Generate reports
    reporter = EvaluationReporter(
        results=results,
        summary=summary,
        output_dir="./evaluation_reports"
    )
    
    report_paths = reporter.generate_all_reports()
    
    print(f"\nGenerated Reports:")
    for fmt, path in report_paths.items():
        print(f"  {fmt.upper()}: {path}")
    
    # Disable instrumentation
    enable_instrumentation(False)
    """
    
    return None


def example_generate_mock_report():
    """Example: Generate a report with mock data."""
    print("\n" + "=" * 60)
    print("Example 5: Generate Mock Report")
    print("=" * 60)
    
    from src.evaluation.evaluator import EvaluationResult, EvaluationSummary
    
    # Create mock results
    mock_results = [
        EvaluationResult(
            sample_id="mock_001",
            question="What is SQL injection?",
            ground_truth="SQL injection is a code injection technique.",
            predicted_answer="SQL injection exploits database vulnerabilities.",
            speed_metrics={"total_execution_time_seconds": 5.2},
            accuracy_metrics={
                "semantic_similarity": 0.85,
                "rouge_scores": {"rouge1": 0.6, "rouge2": 0.4, "rougeL": 0.55},
                "bleu_score": 0.35,
            },
            relevance_metrics={
                "answer_relevance": 0.8,
                "context_relevance": {"mean": 0.75},
            },
            faithfulness_metrics={
                "faithfulness_score": 0.9,
                "hallucination_detection": {"hallucination_score": 0.1},
            },
        ),
        EvaluationResult(
            sample_id="mock_002",
            question="Show login failures",
            ground_truth="Multiple login failures detected.",
            predicted_answer="Login failures found for users admin and root.",
            speed_metrics={"total_execution_time_seconds": 8.5},
            accuracy_metrics={
                "semantic_similarity": 0.72,
                "rouge_scores": {"rouge1": 0.5, "rouge2": 0.3, "rougeL": 0.45},
                "bleu_score": 0.25,
            },
            relevance_metrics={
                "answer_relevance": 0.7,
                "context_relevance": {"mean": 0.65},
            },
            faithfulness_metrics={
                "faithfulness_score": 0.85,
                "hallucination_detection": {"hallucination_score": 0.15},
            },
        ),
    ]
    
    # Create mock summary
    mock_summary = EvaluationSummary(
        total_samples=2,
        successful_evaluations=2,
        failed_evaluations=0,
        aggregate_speed={
            "mean_total_time": 6.85,
            "median_total_time": 6.85,
            "std_total_time": 1.65,
            "min_total_time": 5.2,
            "max_total_time": 8.5,
        },
        aggregate_accuracy={
            "mean_semantic_similarity": 0.785,
            "mean_rouge1": 0.55,
            "mean_bleu": 0.30,
        },
        aggregate_relevance={
            "mean_answer_relevance": 0.75,
            "mean_context_relevance": 0.70,
        },
        aggregate_faithfulness={
            "mean_faithfulness": 0.875,
            "min_faithfulness": 0.85,
        },
    )
    
    # Generate reports
    output_dir = Path("/tmp/mock_evaluation_reports")
    reporter = EvaluationReporter(
        results=mock_results,
        summary=mock_summary,
        output_dir=output_dir,
    )
    
    # Generate markdown report (most readable for console)
    md_path = reporter.generate_markdown_report()
    print(f"\nGenerated Markdown report: {md_path}")
    
    # Read and display a portion
    with open(md_path, "r") as f:
        content = f.read()
        # Show first 2000 characters
        print("\n--- Report Preview (first 2000 chars) ---")
        print(content[:2000])
    
    # Also generate HTML
    html_path = reporter.generate_html_report()
    print(f"\nGenerated HTML report: {html_path}")
    print("(Open in browser to view)")


async def main():
    """Run all examples."""
    print("\n" + "#" * 60)
    print("# Multi-Agent RAG Evaluation System Examples")
    print("#" * 60)
    
    # Example 1: Direct metrics calculation
    example_metrics_calculation()
    
    # Example 2: Dataset operations
    example_dataset_operations()
    
    # Example 3: Load sample dataset
    dataset = example_load_sample_dataset()
    
    # Example 4: Run evaluation (requires environment)
    if dataset:
        await example_run_evaluation(dataset)
    
    # Example 5: Generate mock report
    example_generate_mock_report()
    
    print("\n" + "#" * 60)
    print("# Examples Complete!")
    print("#" * 60)


if __name__ == "__main__":
    asyncio.run(main())
