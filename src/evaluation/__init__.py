# src/evaluation/__init__.py
"""
Evaluation module for the multi-agent RAG system.
Provides tools to measure speed, accuracy, and relevance metrics.
"""

from src.evaluation.metrics import (
    calculate_speed_metrics,
    calculate_accuracy_metrics,
    calculate_relevance_metrics,
    calculate_faithfulness,
)
from src.evaluation.dataset import EvaluationDataset, EvaluationSample
from src.evaluation.evaluator import Evaluator
from src.evaluation.reporter import EvaluationReporter

__all__ = [
    "calculate_speed_metrics",
    "calculate_accuracy_metrics",
    "calculate_relevance_metrics",
    "calculate_faithfulness",
    "EvaluationDataset",
    "EvaluationSample",
    "Evaluator",
    "EvaluationReporter",
]
