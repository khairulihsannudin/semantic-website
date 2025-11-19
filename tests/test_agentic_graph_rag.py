"""
Unit tests for Agentic Graph RAG
"""

import pytest
from src.rag.agentic_graph_rag import AgenticGraphRAG


class TestAgenticGraphRAG:
    """Test cases for Agentic Graph RAG."""
    
    def test_initialization(self):
        """Test RAG initialization."""
        rag = AgenticGraphRAG()
        
        assert rag.encoder is not None
        assert rag.documents == []
        assert rag.kg is not None
        assert rag.kg.graph.number_of_nodes() > 0
    
    def test_add_documents(self):
        """Test adding documents."""
        rag = AgenticGraphRAG()
        
        docs = [
            "This is a test document about phishing.",
            "This document discusses ransomware.",
            "SQL injection is a common vulnerability."
        ]
        
        rag.add_documents(docs)
        
        assert len(rag.documents) == 3
        assert rag.embeddings is not None
        assert rag.index is not None
    
    def test_extract_entities(self):
        """Test entity extraction."""
        rag = AgenticGraphRAG()
        
        text = "Phishing and ransomware are major threats that require multi-factor authentication."
        entities = rag._extract_entities(text)
        
        assert len(entities) > 0
        assert any('phishing' in e.lower() for e in entities)
    
    def test_expand_with_kg(self):
        """Test KG expansion."""
        rag = AgenticGraphRAG()
        
        query = "How to prevent phishing attacks?"
        retrieved_docs = [
            {"document": "Phishing is a threat that can be mitigated with training.", "rank": 1, "score": 0.1}
        ]
        
        kg_context = rag._expand_with_kg(query, retrieved_docs)
        
        assert 'identified_entities' in kg_context
        assert 'entity_contexts' in kg_context
        assert 'mitigations' in kg_context
    
    def test_retrieve(self):
        """Test document retrieval."""
        rag = AgenticGraphRAG()
        
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
    
    def test_get_statistics(self):
        """Test getting statistics."""
        rag = AgenticGraphRAG()
        
        docs = ["Doc 1", "Doc 2", "Doc 3"]
        rag.add_documents(docs)
        
        stats = rag.get_statistics()
        
        assert 'num_documents' in stats
        assert 'kg_statistics' in stats
        assert stats['num_documents'] == 3
        assert stats['kg_statistics']['num_nodes'] > 0
