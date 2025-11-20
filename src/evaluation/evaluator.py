"""
Evaluation Module
Metrics and evaluation framework for comparing RAG approaches
"""

from typing import Dict, List
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class RAGEvaluator:
    """Evaluator for comparing RAG system performance."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize evaluator.
        
        Args:
            model_name: Sentence transformer model for semantic similarity
        """
        self.encoder = SentenceTransformer(model_name)
    
    def evaluate_response(
        self,
        query: str,
        response: str,
        ground_truth: str,
        retrieved_docs: List[Dict]
    ) -> Dict[str, float]:
        """
        Evaluate a single response.
        
        Args:
            query: The query
            response: Generated response
            ground_truth: Ground truth answer
            retrieved_docs: Retrieved documents
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Semantic similarity between response and ground truth
        if ground_truth:
            response_emb = self.encoder.encode([response])
            ground_truth_emb = self.encoder.encode([ground_truth])
            similarity = cosine_similarity(response_emb, ground_truth_emb)[0][0]
            metrics['semantic_similarity'] = float(similarity)
        
        # Retrieval metrics
        metrics['num_retrieved'] = len(retrieved_docs)
        if retrieved_docs:
            metrics['avg_retrieval_score'] = np.mean([doc['similarity'] for doc in retrieved_docs])
            metrics['top_retrieval_score'] = retrieved_docs[0]['similarity'] if retrieved_docs else 0.0
        
        # Response length
        metrics['response_length'] = len(response.split())
        
        return metrics
    
    def evaluate_batch(
        self,
        queries: List[str],
        responses: List[str],
        ground_truths: List[str],
        retrieved_docs_list: List[List[Dict]],
        times: List[float]
    ) -> Dict[str, any]:
        """
        Evaluate a batch of queries.
        
        Args:
            queries: List of queries
            responses: List of responses
            ground_truths: List of ground truth answers
            retrieved_docs_list: List of retrieved documents for each query
            times: List of response times
            
        Returns:
            Aggregated metrics
        """
        individual_metrics = []
        
        for query, response, gt, docs in zip(queries, responses, ground_truths, retrieved_docs_list):
            metrics = self.evaluate_response(query, response, gt, docs)
            individual_metrics.append(metrics)
        
        # Aggregate metrics
        aggregated = {
            'num_queries': len(queries),
            'avg_semantic_similarity': np.mean([m.get('semantic_similarity', 0) for m in individual_metrics]),
            'avg_response_time': np.mean(times) if times else 0.0,
            'avg_num_retrieved': np.mean([m['num_retrieved'] for m in individual_metrics]),
            'avg_retrieval_score': np.mean([m.get('avg_retrieval_score', 0) for m in individual_metrics]),
            'avg_response_length': np.mean([m['response_length'] for m in individual_metrics]),
            'total_time': sum(times) if times else 0.0
        }
        
        return {
            'aggregated': aggregated,
            'individual': individual_metrics
        }
    
    def compare_methods(
        self,
        results_dict: Dict[str, Dict]
    ) -> Dict:
        """
        Compare multiple RAG methods.
        
        Args:
            results_dict: Dictionary mapping method names to their evaluation results
            
        Returns:
            Comparison summary
        """
        comparison = {
            'methods': list(results_dict.keys()),
            'metrics': {}
        }
        
        # Extract key metrics for comparison
        for method, results in results_dict.items():
            agg = results.get('aggregated', {})
            comparison['metrics'][method] = {
                'semantic_similarity': agg.get('avg_semantic_similarity', 0),
                'response_time': agg.get('avg_response_time', 0),
                'retrieval_score': agg.get('avg_retrieval_score', 0),
                'response_length': agg.get('avg_response_length', 0)
            }
        
        # Determine winners for each metric
        comparison['winners'] = {}
        
        if comparison['metrics']:
            # Semantic similarity (higher is better)
            max_sim = max(m.get('semantic_similarity', 0) for m in comparison['metrics'].values())
            comparison['winners']['semantic_similarity'] = [
                method for method, metrics in comparison['metrics'].items()
                if metrics.get('semantic_similarity', 0) == max_sim
            ][0] if max_sim > 0 else None
            
            # Response time (lower is better)
            min_time = min(m.get('response_time', float('inf')) for m in comparison['metrics'].values())
            comparison['winners']['response_time'] = [
                method for method, metrics in comparison['metrics'].items()
                if metrics.get('response_time', float('inf')) == min_time
            ][0] if min_time < float('inf') else None
            
            # Retrieval score (higher is better)
            max_ret = max(m.get('retrieval_score', 0) for m in comparison['metrics'].values())
            comparison['winners']['retrieval_score'] = [
                method for method, metrics in comparison['metrics'].items()
                if metrics.get('retrieval_score', 0) == max_ret
            ][0] if max_ret > 0 else None
        
        return comparison
