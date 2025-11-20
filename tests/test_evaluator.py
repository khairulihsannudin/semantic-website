"""
Unit tests for Evaluator
"""

import pytest
from src.evaluation.evaluator import RAGEvaluator


class TestRAGEvaluator:
    """Test cases for RAG evaluator."""
    
    def test_initialization(self):
        """Test evaluator initialization."""
        evaluator = RAGEvaluator()
        
        assert evaluator.encoder is not None
    
    def test_evaluate_response(self):
        """Test evaluating a single response."""
        evaluator = RAGEvaluator()
        
        query = "What is phishing?"
        response = "Phishing is a social engineering attack."
        ground_truth = "Phishing is an attack that tricks users."
        retrieved_docs = [
            {"document": "Doc 1", "similarity": 0.8, "rank": 1},
            {"document": "Doc 2", "similarity": 0.6, "rank": 2}
        ]
        
        metrics = evaluator.evaluate_response(query, response, ground_truth, retrieved_docs)
        
        assert 'semantic_similarity' in metrics
        assert 'num_retrieved' in metrics
        assert 'avg_retrieval_score' in metrics
        assert 'response_length' in metrics
        assert metrics['num_retrieved'] == 2
    
    def test_evaluate_batch(self):
        """Test batch evaluation."""
        evaluator = RAGEvaluator()
        
        queries = ["Query 1", "Query 2"]
        responses = ["Response 1", "Response 2"]
        ground_truths = ["GT 1", "GT 2"]
        retrieved_docs_list = [
            [{"document": "Doc", "similarity": 0.7, "rank": 1}],
            [{"document": "Doc", "similarity": 0.8, "rank": 1}]
        ]
        times = [1.0, 1.5]
        
        results = evaluator.evaluate_batch(queries, responses, ground_truths, retrieved_docs_list, times)
        
        assert 'aggregated' in results
        assert 'individual' in results
        assert 'avg_semantic_similarity' in results['aggregated']
        assert 'avg_response_time' in results['aggregated']
        assert len(results['individual']) == 2
    
    def test_compare_methods(self):
        """Test comparing methods."""
        evaluator = RAGEvaluator()
        
        results_dict = {
            'Method 1': {
                'aggregated': {
                    'avg_semantic_similarity': 0.75,
                    'avg_response_time': 1.0,
                    'avg_retrieval_score': 0.7,
                    'avg_response_length': 100
                }
            },
            'Method 2': {
                'aggregated': {
                    'avg_semantic_similarity': 0.80,
                    'avg_response_time': 1.5,
                    'avg_retrieval_score': 0.75,
                    'avg_response_length': 120
                }
            }
        }
        
        comparison = evaluator.compare_methods(results_dict)
        
        assert 'methods' in comparison
        assert 'metrics' in comparison
        assert 'winners' in comparison
        assert len(comparison['methods']) == 2
