#!/usr/bin/env python3
# src/evaluation/run_evaluation.py
"""
CLI script to run evaluation on the multi-agent RAG system.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import yaml

from src.evaluation.dataset import EvaluationDataset
from src.evaluation.evaluator import Evaluator
from src.evaluation.reporter import EvaluationReporter
from src.utils.logging_config import setup_logging


logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str | Path]) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    default_config = {
        "evaluation": {
            "batch_size": 1,
            "max_iterations": 3,
        },
        "output": {
            "directory": "./evaluation_reports",
            "formats": ["json", "csv", "html", "markdown"],
        },
        "metrics": {
            "speed": True,
            "accuracy": True,
            "relevance": True,
            "faithfulness": True,
        },
        "thresholds": {
            "min_semantic_similarity": 0.5,
            "min_faithfulness": 0.5,
            "max_execution_time": 60.0,
        },
    }
    
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    # Merge file config with defaults
                    for key, value in file_config.items():
                        if isinstance(value, dict) and key in default_config:
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
    
    return default_config


async def run_evaluation(
    dataset_path: str,
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """
    Run evaluation on the specified dataset.
    
    Args:
        dataset_path: Path to evaluation dataset (JSON or CSV)
        config_path: Optional path to configuration file
        output_dir: Optional output directory for reports
        verbose: Enable verbose logging
    """
    # Setup logging
    setup_logging()
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting evaluation...")
    
    # Load configuration
    config = load_config(config_path)
    logger.info(f"Configuration loaded: batch_size={config['evaluation']['batch_size']}")
    
    # Override output directory if provided
    if output_dir:
        config["output"]["directory"] = output_dir
    
    # Load dataset
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        logger.error(f"Dataset file not found: {dataset_path}")
        sys.exit(1)
    
    logger.info(f"Loading dataset from: {dataset_path}")
    
    if dataset_path.suffix == ".json":
        dataset = EvaluationDataset.from_json(dataset_path)
    elif dataset_path.suffix == ".csv":
        dataset = EvaluationDataset.from_csv(dataset_path)
    else:
        logger.error(f"Unsupported dataset format: {dataset_path.suffix}")
        sys.exit(1)
    
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    logger.info(f"Dataset summary: {dataset.summary()}")
    
    # Create evaluator
    evaluator = Evaluator(config=config)
    
    # Run evaluation
    logger.info("Running evaluation...")
    try:
        results = await evaluator.evaluate_dataset(
            dataset,
            batch_size=config["evaluation"]["batch_size"],
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)
    
    # Generate summary
    logger.info("Generating summary...")
    summary = evaluator.generate_summary(dataset)
    
    # Generate reports
    logger.info(f"Generating reports to: {config['output']['directory']}")
    reporter = EvaluationReporter(
        results=results,
        summary=summary,
        output_dir=config["output"]["directory"],
    )
    
    report_paths = {}
    formats = config["output"]["formats"]
    
    if "json" in formats:
        report_paths["json"] = reporter.generate_json_report()
    if "csv" in formats:
        report_paths["csv"] = reporter.generate_csv_report()
    if "html" in formats:
        report_paths["html"] = reporter.generate_html_report()
    if "markdown" in formats:
        report_paths["markdown"] = reporter.generate_markdown_report()
    
    # Print summary to console
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nTotal Samples: {summary.total_samples}")
    print(f"Successful: {summary.successful_evaluations}")
    print(f"Failed: {summary.failed_evaluations}")
    print(f"Success Rate: {summary.successful_evaluations / max(summary.total_samples, 1) * 100:.1f}%")
    
    print("\n--- Aggregate Metrics ---")
    print(f"Mean Execution Time: {summary.aggregate_speed.get('mean_total_time', 0):.2f}s")
    print(f"Mean Semantic Similarity: {summary.aggregate_accuracy.get('mean_semantic_similarity', 0):.4f}")
    print(f"Mean Answer Relevance: {summary.aggregate_relevance.get('mean_answer_relevance', 0):.4f}")
    print(f"Mean Faithfulness: {summary.aggregate_faithfulness.get('mean_faithfulness', 0):.4f}")
    
    # Check thresholds
    print("\n--- Threshold Checks ---")
    thresholds = config["thresholds"]
    
    sem_sim = summary.aggregate_accuracy.get("mean_semantic_similarity", 0)
    faithful = summary.aggregate_faithfulness.get("mean_faithfulness", 0)
    exec_time = summary.aggregate_speed.get("mean_total_time", 0)
    
    sem_sim_pass = sem_sim >= thresholds["min_semantic_similarity"]
    faithful_pass = faithful >= thresholds["min_faithfulness"]
    time_pass = exec_time <= thresholds["max_execution_time"]
    
    print(f"Semantic Similarity >= {thresholds['min_semantic_similarity']}: {'✅ PASS' if sem_sim_pass else '❌ FAIL'}")
    print(f"Faithfulness >= {thresholds['min_faithfulness']}: {'✅ PASS' if faithful_pass else '❌ FAIL'}")
    print(f"Execution Time <= {thresholds['max_execution_time']}s: {'✅ PASS' if time_pass else '❌ FAIL'}")
    
    print("\n--- Generated Reports ---")
    for fmt, path in report_paths.items():
        print(f"  {fmt.upper()}: {path}")
    
    print("=" * 60 + "\n")
    
    # Exit with error code if thresholds not met
    if not all([sem_sim_pass, faithful_pass, time_pass]):
        logger.warning("Some evaluation thresholds were not met.")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Run evaluation on the multi-agent RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evaluation with default settings
  python -m src.evaluation.run_evaluation data/evaluation/sample_eval_dataset.json

  # Run with custom config and output directory
  python -m src.evaluation.run_evaluation data/evaluation/sample_eval_dataset.json \\
      --config config/evaluation_config.yaml \\
      --output ./my_reports

  # Run with verbose logging
  python -m src.evaluation.run_evaluation data/evaluation/sample_eval_dataset.json --verbose
        """,
    )
    
    parser.add_argument(
        "dataset",
        type=str,
        help="Path to evaluation dataset (JSON or CSV)",
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to configuration YAML file",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for reports",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Run the async evaluation
    asyncio.run(run_evaluation(
        dataset_path=args.dataset,
        config_path=args.config,
        output_dir=args.output,
        verbose=args.verbose,
    ))


if __name__ == "__main__":
    main()
