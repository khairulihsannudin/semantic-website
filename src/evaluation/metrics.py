# src/evaluation/metrics.py
"""
Individual metric calculators for evaluating the multi-agent RAG system.
"""

import re
from typing import Any, Optional
from collections import Counter
import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer


# Initialize sentence transformer for semantic similarity
_embedding_model: Optional[SentenceTransformer] = None


def _get_embedding_model() -> SentenceTransformer:
    """Get or initialize the embedding model (lazy loading)."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedding_model


def calculate_speed_metrics(execution_data: dict[str, Any]) -> dict[str, Any]:
    """
    Parse timing data from workflow execution.
    
    Args:
        execution_data: Dictionary containing timing information from instrumented workflow
        
    Returns:
        Dictionary with speed metrics including:
        - total_execution_time: End-to-end query processing time in seconds
        - per_node_times: Time spent in each agent node
        - retrieval_times: Separate timing for vector/cypher operations
        - latency_breakdown: Detailed timing for each step
    """
    timing_data = execution_data.get("timing_data", {})
    
    # Calculate total execution time
    total_time = timing_data.get("total_time", 0.0)
    
    # Extract per-node times
    node_times = timing_data.get("node_times", {})
    
    # Calculate retrieval times (vector and cypher)
    retrieval_times = {
        "vector_search": node_times.get("vector_agent", 0.0),
        "cypher_query": node_times.get("cypher_agent", 0.0),
        "mcp_rdf": node_times.get("mcp_rdf_agent", 0.0),
    }
    
    # Calculate aggregate statistics
    node_time_values = list(node_times.values()) if node_times else [0.0]
    
    return {
        "total_execution_time_seconds": total_time,
        "per_node_times": node_times,
        "retrieval_times": retrieval_times,
        "latency_breakdown": {
            "guardrails_overhead": node_times.get("guardrails", 0.0),
            "vector_pipeline": (
                node_times.get("vector_agent", 0.0) +
                node_times.get("review_vector_answer", 0.0) +
                node_times.get("vector_reflection", 0.0)
            ),
            "cypher_pipeline": (
                node_times.get("cypher_agent", 0.0) +
                node_times.get("review_cypher_answer", 0.0) +
                node_times.get("cypher_reflection", 0.0)
            ),
            "analysis_synthesis": (
                node_times.get("log_analysis_agent", 0.0) +
                node_times.get("mcp_rdf_agent", 0.0) +
                node_times.get("synthesizer", 0.0)
            ),
        },
        "statistics": {
            "mean_node_time": float(np.mean(node_time_values)),
            "max_node_time": float(np.max(node_time_values)),
            "min_node_time": float(np.min(node_time_values)),
        },
        "execution_path": timing_data.get("execution_path", []),
        "iterations": {
            "vector": timing_data.get("vector_iterations", 0),
            "cypher": timing_data.get("cypher_iterations", 0),
        },
    }


def _tokenize(text: str) -> list[str]:
    """Simple word tokenization."""
    return re.findall(r'\b\w+\b', text.lower())


def _calculate_exact_match(predicted: str, ground_truth: str) -> float:
    """Calculate exact match score (0 or 1)."""
    return 1.0 if predicted.strip().lower() == ground_truth.strip().lower() else 0.0


def _calculate_semantic_similarity(predicted: str, ground_truth: str) -> float:
    """Calculate semantic similarity using sentence embeddings."""
    if not predicted or not ground_truth:
        return 0.0
    
    model = _get_embedding_model()
    embeddings = model.encode([predicted, ground_truth])
    
    # Cosine similarity
    similarity = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    )
    return float(similarity)


def _calculate_rouge_scores(predicted: str, ground_truth: str) -> dict[str, float]:
    """Calculate ROUGE scores."""
    if not predicted or not ground_truth:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(ground_truth, predicted)
    
    return {
        "rouge1": scores['rouge1'].fmeasure,
        "rouge2": scores['rouge2'].fmeasure,
        "rougeL": scores['rougeL'].fmeasure,
    }


def _calculate_bleu_score(predicted: str, ground_truth: str) -> float:
    """Calculate BLEU score."""
    if not predicted or not ground_truth:
        return 0.0
    
    reference = [_tokenize(ground_truth)]
    candidate = _tokenize(predicted)
    
    if not candidate:
        return 0.0
    
    # Use smoothing function for short sentences
    smoothing = SmoothingFunction().method1
    try:
        return sentence_bleu(reference, candidate, smoothing_function=smoothing)
    except (ZeroDivisionError, ValueError):
        return 0.0


def calculate_accuracy_metrics(predicted: str, ground_truth: str) -> dict[str, Any]:
    """
    Compare generated answers against ground truth.
    
    Args:
        predicted: The generated answer
        ground_truth: The expected correct answer
        
    Returns:
        Dictionary with accuracy metrics including:
        - exact_match: Binary score for exact match
        - semantic_similarity: Cosine similarity of embeddings
        - rouge_scores: ROUGE-1, ROUGE-2, ROUGE-L scores
        - bleu_score: BLEU score
    """
    return {
        "exact_match": _calculate_exact_match(predicted, ground_truth),
        "semantic_similarity": _calculate_semantic_similarity(predicted, ground_truth),
        "rouge_scores": _calculate_rouge_scores(predicted, ground_truth),
        "bleu_score": _calculate_bleu_score(predicted, ground_truth),
    }


def _calculate_answer_relevance(question: str, answer: str) -> float:
    """Calculate how relevant the answer is to the question."""
    if not question or not answer:
        return 0.0
    
    return _calculate_semantic_similarity(question, answer)


def _calculate_context_relevance(question: str, contexts: list[str]) -> dict[str, Any]:
    """Calculate relevance of retrieved contexts to the question."""
    if not question or not contexts:
        return {"mean": 0.0, "max": 0.0, "min": 0.0, "individual": []}
    
    scores = [_calculate_semantic_similarity(question, ctx) for ctx in contexts if ctx]
    
    if not scores:
        return {"mean": 0.0, "max": 0.0, "min": 0.0, "individual": []}
    
    return {
        "mean": float(np.mean(scores)),
        "max": float(np.max(scores)),
        "min": float(np.min(scores)),
        "individual": scores,
    }


def calculate_relevance_metrics(
    question: str,
    answer: str,
    contexts: list[str],
) -> dict[str, Any]:
    """
    Measure relevance of answers and contexts.
    
    Args:
        question: The original question
        answer: The generated answer
        contexts: List of retrieved contexts (from vector, cypher, MCP RDF)
        
    Returns:
        Dictionary with relevance metrics including:
        - answer_relevance: How relevant is the answer to the question
        - context_relevance: How relevant are the retrieved contexts
    """
    return {
        "answer_relevance": _calculate_answer_relevance(question, answer),
        "context_relevance": _calculate_context_relevance(question, contexts),
    }


def _check_hallucination(answer: str, contexts: list[str]) -> dict[str, Any]:
    """
    Check if the answer contains information not present in contexts.
    Uses keyword overlap and semantic similarity.
    """
    if not answer or not contexts:
        return {
            "hallucination_score": 1.0 if answer else 0.0,
            "unsupported_ratio": 1.0 if answer else 0.0,
        }
    
    # Combine all contexts
    combined_context = " ".join(contexts)
    
    # Tokenize answer and context
    answer_tokens = set(_tokenize(answer))
    context_tokens = set(_tokenize(combined_context))
    
    # Calculate token overlap
    common_tokens = answer_tokens & context_tokens
    unsupported_tokens = answer_tokens - context_tokens
    
    # Filter out common stopwords for better detection
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'and', 'or', 'but',
        'if', 'then', 'else', 'when', 'where', 'which', 'who', 'whom', 'this',
        'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
        'it', 'its', 'of', 'for', 'with', 'as', 'at', 'by', 'to', 'from',
        'in', 'on', 'not', 'no', 'yes', 'so', 'up', 'down', 'out', 'into',
    }
    
    meaningful_answer_tokens = answer_tokens - stopwords
    meaningful_unsupported = unsupported_tokens - stopwords
    
    if not meaningful_answer_tokens:
        unsupported_ratio = 0.0
    else:
        unsupported_ratio = len(meaningful_unsupported) / len(meaningful_answer_tokens)
    
    # Calculate semantic similarity between answer and combined context
    semantic_grounding = _calculate_semantic_similarity(answer, combined_context)
    
    # Hallucination score: higher means more hallucination
    # Combining token-based and semantic measures
    hallucination_score = 0.5 * unsupported_ratio + 0.5 * (1 - semantic_grounding)
    
    return {
        "hallucination_score": hallucination_score,
        "unsupported_ratio": unsupported_ratio,
        "semantic_grounding": semantic_grounding,
        "supported_tokens": len(common_tokens - stopwords),
        "unsupported_tokens": len(meaningful_unsupported),
    }


def calculate_faithfulness(answer: str, contexts: list[str]) -> dict[str, Any]:
    """
    Measure if the answer is faithful to the retrieved contexts.
    
    Args:
        answer: The generated answer
        contexts: List of retrieved contexts
        
    Returns:
        Dictionary with faithfulness metrics including:
        - faithfulness_score: How faithful the answer is to contexts (0-1, higher is better)
        - hallucination_detection: Detailed hallucination analysis
    """
    hallucination_result = _check_hallucination(answer, contexts)
    
    # Faithfulness is inverse of hallucination
    faithfulness_score = 1.0 - hallucination_result["hallucination_score"]
    
    return {
        "faithfulness_score": faithfulness_score,
        "hallucination_detection": hallucination_result,
    }


def calculate_context_precision(
    retrieved_contexts: list[str],
    relevant_contexts: list[str],
) -> float:
    """
    Measure if retrieved contexts contain relevant information.
    
    Args:
        retrieved_contexts: Contexts actually retrieved
        relevant_contexts: Expected relevant contexts (ground truth)
        
    Returns:
        Precision score (0-1)
    """
    if not retrieved_contexts or not relevant_contexts:
        return 0.0
    
    # Calculate semantic similarity between retrieved and relevant contexts
    relevant_count = 0
    threshold = 0.5  # Similarity threshold to consider a context as relevant
    
    for retrieved in retrieved_contexts:
        for relevant in relevant_contexts:
            similarity = _calculate_semantic_similarity(retrieved, relevant)
            if similarity >= threshold:
                relevant_count += 1
                break
    
    return relevant_count / len(retrieved_contexts)


def calculate_context_recall(
    retrieved_contexts: list[str],
    relevant_contexts: list[str],
) -> float:
    """
    Measure if all relevant information was retrieved.
    
    Args:
        retrieved_contexts: Contexts actually retrieved
        relevant_contexts: Expected relevant contexts (ground truth)
        
    Returns:
        Recall score (0-1)
    """
    if not relevant_contexts:
        return 1.0  # If no relevant contexts expected, recall is perfect
    
    if not retrieved_contexts:
        return 0.0
    
    # Calculate how many relevant contexts were retrieved
    retrieved_count = 0
    threshold = 0.5  # Similarity threshold
    
    for relevant in relevant_contexts:
        for retrieved in retrieved_contexts:
            similarity = _calculate_semantic_similarity(relevant, retrieved)
            if similarity >= threshold:
                retrieved_count += 1
                break
    
    return retrieved_count / len(relevant_contexts)
