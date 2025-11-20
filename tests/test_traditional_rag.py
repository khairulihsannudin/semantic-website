"""
Unit tests for Traditional RAG
"""

import pytest
from src.rag.traditional_rag import TraditionalRAG


class TestTraditionalRAG:
    """Test cases for Traditional RAG."""
    
    def test_initialization(self):
        """Test RAG initialization."""
        rag = TraditionalRAG()
        
        assert rag.encoder is not None
        assert rag.documents == []
        assert rag.embeddings is None
        assert rag.index is None
    
    def test_add_documents(self):
        """Test adding documents."""
        rag = TraditionalRAG()
        
        docs = [
            "This is a test document about phishing.",
            "This document discusses ransomware.",
            "SQL injection is a common vulnerability."
        ]
        
        rag.add_documents(docs)
        
        assert len(rag.documents) == 3
        assert rag.embeddings is not None
        assert rag.index is not None
        assert rag.dimension is not None
    
    def test_retrieve(self):
        """Test document retrieval."""
        rag = TraditionalRAG()
        
        docs = [
            "Phishing is a social engineering attack.",
            "Ransomware encrypts files and demands payment.",
            "SQL injection exploits database vulnerabilities."
        ]
        
        rag.add_documents(docs)
        
        query = "What is phishing?"
        results = rag.retrieve(query, top_k=2)
        
        assert len(results) <= 2
        assert len(results) > 0
        assert 'document' in results[0]
        assert 'score' in results[0]
        assert 'rank' in results[0]
    
    def test_retrieve_empty(self):
        """Test retrieval with no documents."""
        rag = TraditionalRAG()
        
        results = rag.retrieve("test query", top_k=5)
        
        assert results == []
    
    def test_get_statistics(self):
        """Test getting statistics."""
        rag = TraditionalRAG()
        
        docs = ["Doc 1", "Doc 2", "Doc 3"]
        rag.add_documents(docs)
        
        stats = rag.get_statistics()
        
        assert 'num_documents' in stats
        assert 'embedding_dimension' in stats
        assert stats['num_documents'] == 3
        assert stats['embedding_dimension'] > 0
