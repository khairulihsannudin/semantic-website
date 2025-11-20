"""
Unit tests for Cybersecurity Knowledge Graph
"""

import pytest
from src.knowledge_graph.cskg import CybersecurityKnowledgeGraph


class TestCybersecurityKnowledgeGraph:
    """Test cases for CSKG."""
    
    def test_initialization(self):
        """Test KG initialization."""
        kg = CybersecurityKnowledgeGraph()
        
        assert kg.graph is not None
        assert kg.graph.number_of_nodes() > 0
        assert kg.graph.number_of_edges() > 0
    
    def test_graph_statistics(self):
        """Test graph statistics."""
        kg = CybersecurityKnowledgeGraph()
        stats = kg.get_graph_statistics()
        
        assert 'num_nodes' in stats
        assert 'num_edges' in stats
        assert 'node_types' in stats
        assert stats['num_nodes'] > 0
        assert stats['num_edges'] > 0
    
    def test_get_all_entities_by_type(self):
        """Test getting entities by type."""
        kg = CybersecurityKnowledgeGraph()
        
        threats = kg.get_all_entities_by_type('threat')
        mitigations = kg.get_all_entities_by_type('mitigation')
        vulnerabilities = kg.get_all_entities_by_type('vulnerability')
        
        assert len(threats) > 0
        assert len(mitigations) > 0
        assert len(vulnerabilities) > 0
        assert 'Phishing' in threats
        assert 'Multi-Factor Authentication' in mitigations
    
    def test_query_related_entities(self):
        """Test querying related entities."""
        kg = CybersecurityKnowledgeGraph()
        
        related = kg.query_related_entities('Phishing', max_depth=2)
        
        assert 'direct' in related
        assert 'indirect' in related
        assert len(related['direct']) > 0
    
    def test_get_entity_context(self):
        """Test getting entity context."""
        kg = CybersecurityKnowledgeGraph()
        
        context = kg.get_entity_context('Malware')
        
        assert context is not None
        assert 'entity' in context
        assert 'attributes' in context
        assert 'related' in context
        assert context['entity'] == 'Malware'
    
    def test_find_mitigation_path(self):
        """Test finding mitigation paths."""
        kg = CybersecurityKnowledgeGraph()
        
        mitigations = kg.find_mitigation_path('Phishing')
        
        assert isinstance(mitigations, list)
        # Should find at least one mitigation for phishing
        assert len(mitigations) > 0
    
    def test_export_to_dict(self):
        """Test exporting KG to dictionary."""
        kg = CybersecurityKnowledgeGraph()
        
        data = kg.export_to_dict()
        
        assert 'nodes' in data
        assert 'edges' in data
        assert len(data['nodes']) > 0
        assert len(data['edges']) > 0
    
    def test_nonexistent_entity(self):
        """Test querying non-existent entity."""
        kg = CybersecurityKnowledgeGraph()
        
        context = kg.get_entity_context('NonExistentThreat')
        
        assert context is None
