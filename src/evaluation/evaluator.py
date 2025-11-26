# src/evaluation/evaluator.py
"""
Main evaluation orchestrator for the multi-agent RAG system.
"""

import time
import asyncio
import logging
from typing import Any, Optional
from dataclasses import dataclass, field

from src.evaluation.dataset import EvaluationDataset, EvaluationSample
from src.evaluation.metrics import (
    calculate_speed_metrics,
    calculate_accuracy_metrics,
    calculate_relevance_metrics,
    calculate_faithfulness,
    calculate_context_precision,
    calculate_context_recall,
)


logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """
    Result of evaluating a single sample.
    """
    sample_id: str
    question: str
    ground_truth: str
    predicted_answer: str
    speed_metrics: dict[str, Any] = field(default_factory=dict)
    accuracy_metrics: dict[str, Any] = field(default_factory=dict)
    relevance_metrics: dict[str, Any] = field(default_factory=dict)
    faithfulness_metrics: dict[str, Any] = field(default_factory=dict)
    context_metrics: dict[str, Any] = field(default_factory=dict)
    execution_data: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sample_id": self.sample_id,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "predicted_answer": self.predicted_answer,
            "speed_metrics": self.speed_metrics,
            "accuracy_metrics": self.accuracy_metrics,
            "relevance_metrics": self.relevance_metrics,
            "faithfulness_metrics": self.faithfulness_metrics,
            "context_metrics": self.context_metrics,
            "execution_data": self.execution_data,
            "error": self.error,
        }


@dataclass
class EvaluationSummary:
    """
    Aggregated evaluation results.
    """
    total_samples: int
    successful_evaluations: int
    failed_evaluations: int
    aggregate_speed: dict[str, Any] = field(default_factory=dict)
    aggregate_accuracy: dict[str, Any] = field(default_factory=dict)
    aggregate_relevance: dict[str, Any] = field(default_factory=dict)
    aggregate_faithfulness: dict[str, Any] = field(default_factory=dict)
    per_type_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    per_difficulty_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_samples": self.total_samples,
            "successful_evaluations": self.successful_evaluations,
            "failed_evaluations": self.failed_evaluations,
            "aggregate_speed": self.aggregate_speed,
            "aggregate_accuracy": self.aggregate_accuracy,
            "aggregate_relevance": self.aggregate_relevance,
            "aggregate_faithfulness": self.aggregate_faithfulness,
            "per_type_metrics": self.per_type_metrics,
            "per_difficulty_metrics": self.per_difficulty_metrics,
        }


class Evaluator:
    """
    Main evaluation orchestrator.
    Runs queries through the workflow and collects all metrics.
    """
    
    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize the evaluator.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.results: list[EvaluationResult] = []
        self._workflow_app = None
    
    def _get_workflow_app(self):
        """Lazy load the workflow app to avoid circular imports."""
        if self._workflow_app is None:
            from src.graph.workflow import app
            self._workflow_app = app
        return self._workflow_app
    
    async def _run_single_query(
        self,
        question: str,
        max_iterations: int = 3,
    ) -> tuple[str, dict[str, Any]]:
        """
        Run a single query through the workflow.
        
        Args:
            question: The question to ask
            max_iterations: Maximum iterations for reflection loops
            
        Returns:
            Tuple of (answer, execution_data)
        """
        app = self._get_workflow_app()
        
        initial_state = {
            "question": question,
            "original_question": question,
            "messages": [("human", question)],
            "cypher_iteration_count": 1,
            "vector_iteration_count": 1,
            "max_iterations": max_iterations,
        }
        
        config = {
            "recursion_limit": 30,
            "configurable": {"instrumentation_enabled": True},
        }
        
        start_time = time.time()
        
        try:
            result = await app.ainvoke(initial_state, config=config)
            end_time = time.time()
            
            answer = result.get("answer", "")
            
            # Extract execution data including timing information
            execution_data = {
                "timing_data": result.get("_timing_data", {
                    "total_time": end_time - start_time,
                    "node_times": {},
                    "execution_path": [],
                }),
                "contexts": {
                    "vector": result.get("log_vector_context", ""),
                    "cypher": str(result.get("log_cypher_context", "")),
                    "mcp_rdf": result.get("mcp_rdf_context", ""),
                },
                "intermediate": {
                    "is_log_question": result.get("is_log_question", False),
                    "is_cskg_required": result.get("is_cskg_required", False),
                    "generated_question_for_rdf": result.get("generated_question_for_rdf", ""),
                    "cypher_query": result.get("cypher_query", ""),
                },
                "iterations": {
                    "vector": result.get("vector_iteration_count", 1),
                    "cypher": result.get("cypher_iteration_count", 1),
                },
            }
            
            # If timing data was not captured by instrumentation, add basic timing
            if not result.get("_timing_data"):
                execution_data["timing_data"] = {
                    "total_time": end_time - start_time,
                    "node_times": {},
                    "execution_path": [],
                    "vector_iterations": result.get("vector_iteration_count", 1),
                    "cypher_iterations": result.get("cypher_iteration_count", 1),
                }
            
            return answer, execution_data
            
        except Exception as e:
            logger.error(f"Error running query: {e}")
            end_time = time.time()
            return "", {
                "error": str(e),
                "timing_data": {"total_time": end_time - start_time},
                "contexts": {"vector": "", "cypher": "", "mcp_rdf": ""},
            }
    
    def _extract_contexts(self, execution_data: dict[str, Any]) -> list[str]:
        """Extract all contexts from execution data as a list."""
        contexts = []
        ctx_data = execution_data.get("contexts", {})
        
        for key in ["vector", "cypher", "mcp_rdf"]:
            ctx = ctx_data.get(key, "")
            if ctx and ctx not in ["", "None", "[]", "Not applicable for this query."]:
                contexts.append(str(ctx))
        
        return contexts
    
    async def evaluate_sample(
        self,
        sample: EvaluationSample,
    ) -> EvaluationResult:
        """
        Evaluate a single sample.
        
        Args:
            sample: The evaluation sample
            
        Returns:
            EvaluationResult with all metrics
        """
        logger.info(f"Evaluating sample: {sample.id}")
        
        try:
            # Run the query
            predicted_answer, execution_data = await self._run_single_query(sample.question)
            
            # Extract contexts for relevance/faithfulness calculations
            contexts = self._extract_contexts(execution_data)
            
            # Calculate all metrics
            speed_metrics = calculate_speed_metrics(execution_data)
            accuracy_metrics = calculate_accuracy_metrics(predicted_answer, sample.ground_truth_answer)
            relevance_metrics = calculate_relevance_metrics(sample.question, predicted_answer, contexts)
            faithfulness_metrics = calculate_faithfulness(predicted_answer, contexts)
            
            # Calculate context precision/recall if expected contexts provided
            context_metrics = {}
            if sample.expected_contexts:
                context_metrics = {
                    "precision": calculate_context_precision(contexts, sample.expected_contexts),
                    "recall": calculate_context_recall(contexts, sample.expected_contexts),
                }
            
            result = EvaluationResult(
                sample_id=sample.id,
                question=sample.question,
                ground_truth=sample.ground_truth_answer,
                predicted_answer=predicted_answer,
                speed_metrics=speed_metrics,
                accuracy_metrics=accuracy_metrics,
                relevance_metrics=relevance_metrics,
                faithfulness_metrics=faithfulness_metrics,
                context_metrics=context_metrics,
                execution_data=execution_data,
            )
            
        except Exception as e:
            logger.error(f"Error evaluating sample {sample.id}: {e}")
            result = EvaluationResult(
                sample_id=sample.id,
                question=sample.question,
                ground_truth=sample.ground_truth_answer,
                predicted_answer="",
                error=str(e),
            )
        
        self.results.append(result)
        return result
    
    async def evaluate_dataset(
        self,
        dataset: EvaluationDataset,
        batch_size: int = 1,
    ) -> list[EvaluationResult]:
        """
        Evaluate all samples in a dataset.
        
        Args:
            dataset: The evaluation dataset
            batch_size: Number of samples to process concurrently
            
        Returns:
            List of EvaluationResults
        """
        logger.info(f"Starting evaluation of {len(dataset)} samples")
        self.results = []  # Reset results
        
        # Process samples in batches
        samples = list(dataset)
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}/{(len(samples) + batch_size - 1) // batch_size}")
            
            # Process batch concurrently
            tasks = [self.evaluate_sample(sample) for sample in batch]
            await asyncio.gather(*tasks)
        
        logger.info(f"Evaluation complete. {len(self.results)} samples evaluated.")
        return self.results
    
    def _calculate_aggregate_metrics(
        self,
        results: list[EvaluationResult],
    ) -> dict[str, dict[str, Any]]:
        """Calculate aggregate metrics from a list of results."""
        import numpy as np
        
        if not results:
            return {
                "speed": {},
                "accuracy": {},
                "relevance": {},
                "faithfulness": {},
            }
        
        # Filter out failed evaluations
        valid_results = [r for r in results if r.error is None]
        
        if not valid_results:
            return {
                "speed": {},
                "accuracy": {},
                "relevance": {},
                "faithfulness": {},
            }
        
        # Aggregate speed metrics
        total_times = [r.speed_metrics.get("total_execution_time_seconds", 0) for r in valid_results]
        speed_aggregate = {
            "mean_total_time": float(np.mean(total_times)),
            "median_total_time": float(np.median(total_times)),
            "std_total_time": float(np.std(total_times)),
            "min_total_time": float(np.min(total_times)),
            "max_total_time": float(np.max(total_times)),
        }
        
        # Aggregate accuracy metrics
        semantic_similarities = [r.accuracy_metrics.get("semantic_similarity", 0) for r in valid_results]
        rouge1_scores = [r.accuracy_metrics.get("rouge_scores", {}).get("rouge1", 0) for r in valid_results]
        bleu_scores = [r.accuracy_metrics.get("bleu_score", 0) for r in valid_results]
        
        accuracy_aggregate = {
            "mean_semantic_similarity": float(np.mean(semantic_similarities)),
            "mean_rouge1": float(np.mean(rouge1_scores)),
            "mean_bleu": float(np.mean(bleu_scores)),
            "std_semantic_similarity": float(np.std(semantic_similarities)),
        }
        
        # Aggregate relevance metrics
        answer_relevances = [r.relevance_metrics.get("answer_relevance", 0) for r in valid_results]
        context_relevances = [
            r.relevance_metrics.get("context_relevance", {}).get("mean", 0) 
            for r in valid_results
        ]
        
        relevance_aggregate = {
            "mean_answer_relevance": float(np.mean(answer_relevances)),
            "mean_context_relevance": float(np.mean(context_relevances)),
            "std_answer_relevance": float(np.std(answer_relevances)),
        }
        
        # Aggregate faithfulness metrics
        faithfulness_scores = [
            r.faithfulness_metrics.get("faithfulness_score", 0) 
            for r in valid_results
        ]
        
        faithfulness_aggregate = {
            "mean_faithfulness": float(np.mean(faithfulness_scores)),
            "std_faithfulness": float(np.std(faithfulness_scores)),
            "min_faithfulness": float(np.min(faithfulness_scores)),
        }
        
        return {
            "speed": speed_aggregate,
            "accuracy": accuracy_aggregate,
            "relevance": relevance_aggregate,
            "faithfulness": faithfulness_aggregate,
        }
    
    def generate_summary(
        self,
        dataset: Optional[EvaluationDataset] = None,
    ) -> EvaluationSummary:
        """
        Generate aggregated summary from evaluation results.
        
        Args:
            dataset: Optional dataset to compute per-type and per-difficulty metrics
            
        Returns:
            EvaluationSummary with aggregate statistics
        """
        valid_results = [r for r in self.results if r.error is None]
        failed_results = [r for r in self.results if r.error is not None]
        
        aggregates = self._calculate_aggregate_metrics(valid_results)
        
        # Calculate per-type and per-difficulty metrics if dataset provided
        per_type_metrics = {}
        per_difficulty_metrics = {}
        
        if dataset:
            # Group results by type and difficulty
            type_groups: dict[str, list[EvaluationResult]] = {}
            difficulty_groups: dict[str, list[EvaluationResult]] = {}
            
            for result in valid_results:
                # Find corresponding sample
                sample = next(
                    (s for s in dataset if s.id == result.sample_id), 
                    None
                )
                if sample:
                    # Group by type
                    if sample.question_type not in type_groups:
                        type_groups[sample.question_type] = []
                    type_groups[sample.question_type].append(result)
                    
                    # Group by difficulty
                    if sample.difficulty not in difficulty_groups:
                        difficulty_groups[sample.difficulty] = []
                    difficulty_groups[sample.difficulty].append(result)
            
            # Calculate per-group metrics
            for qtype, type_results in type_groups.items():
                per_type_metrics[qtype] = self._calculate_aggregate_metrics(type_results)
            
            for difficulty, diff_results in difficulty_groups.items():
                per_difficulty_metrics[difficulty] = self._calculate_aggregate_metrics(diff_results)
        
        return EvaluationSummary(
            total_samples=len(self.results),
            successful_evaluations=len(valid_results),
            failed_evaluations=len(failed_results),
            aggregate_speed=aggregates["speed"],
            aggregate_accuracy=aggregates["accuracy"],
            aggregate_relevance=aggregates["relevance"],
            aggregate_faithfulness=aggregates["faithfulness"],
            per_type_metrics=per_type_metrics,
            per_difficulty_metrics=per_difficulty_metrics,
        )
