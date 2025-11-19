"""
Cybersecurity Knowledge Graph (CSKG) Module
Implements knowledge graph construction and querying for cybersecurity domain.
"""

import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
import json


class CybersecurityKnowledgeGraph:
    """
    A knowledge graph for cybersecurity concepts including threats, vulnerabilities,
    attack patterns, mitigation strategies, and their relationships.
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self._initialize_cskg()
    
    def _initialize_cskg(self):
        """Initialize the cybersecurity knowledge graph with domain knowledge."""
        
        # Define cybersecurity entities
        threats = [
            ("Malware", {"type": "threat", "severity": "high", "description": "Malicious software designed to damage or disrupt systems"}),
            ("Phishing", {"type": "threat", "severity": "high", "description": "Social engineering attack to steal sensitive information"}),
            ("DDoS", {"type": "threat", "severity": "high", "description": "Distributed Denial of Service attack"}),
            ("SQL Injection", {"type": "threat", "severity": "critical", "description": "Code injection technique targeting databases"}),
            ("XSS", {"type": "threat", "severity": "medium", "description": "Cross-Site Scripting vulnerability"}),
            ("Ransomware", {"type": "threat", "severity": "critical", "description": "Malware that encrypts data and demands ransom"}),
            ("Zero-Day", {"type": "threat", "severity": "critical", "description": "Previously unknown vulnerability"}),
            ("Man-in-the-Middle", {"type": "threat", "severity": "high", "description": "Interception of communication between two parties"}),
        ]
        
        vulnerabilities = [
            ("Buffer Overflow", {"type": "vulnerability", "cvss": 8.5, "description": "Memory corruption vulnerability"}),
            ("Weak Authentication", {"type": "vulnerability", "cvss": 7.0, "description": "Insufficient authentication mechanisms"}),
            ("Unpatched Software", {"type": "vulnerability", "cvss": 7.5, "description": "Software without security updates"}),
            ("Misconfiguration", {"type": "vulnerability", "cvss": 6.5, "description": "Improper system configuration"}),
            ("Insufficient Encryption", {"type": "vulnerability", "cvss": 7.0, "description": "Weak or missing encryption"}),
        ]
        
        mitigations = [
            ("Multi-Factor Authentication", {"type": "mitigation", "effectiveness": "high", "description": "Multiple authentication factors"}),
            ("Patch Management", {"type": "mitigation", "effectiveness": "high", "description": "Regular security updates"}),
            ("Input Validation", {"type": "mitigation", "effectiveness": "high", "description": "Validate and sanitize user input"}),
            ("Encryption", {"type": "mitigation", "effectiveness": "high", "description": "Data encryption at rest and in transit"}),
            ("Firewall", {"type": "mitigation", "effectiveness": "medium", "description": "Network traffic filtering"}),
            ("IDS/IPS", {"type": "mitigation", "effectiveness": "medium", "description": "Intrusion detection and prevention"}),
            ("Security Training", {"type": "mitigation", "effectiveness": "medium", "description": "User security awareness training"}),
        ]
        
        attack_patterns = [
            ("Credential Stuffing", {"type": "attack_pattern", "description": "Automated injection of stolen credentials"}),
            ("Brute Force", {"type": "attack_pattern", "description": "Systematic trial of all possible combinations"}),
            ("Session Hijacking", {"type": "attack_pattern", "description": "Takeover of a user session"}),
            ("Code Injection", {"type": "attack_pattern", "description": "Injection of malicious code"}),
        ]
        
        # Add nodes to graph
        for entity, attrs in threats + vulnerabilities + mitigations + attack_patterns:
            self.graph.add_node(entity, **attrs)
        
        # Define relationships
        relationships = [
            # Threats exploit vulnerabilities
            ("Malware", "exploits", "Unpatched Software"),
            ("Phishing", "exploits", "Weak Authentication"),
            ("SQL Injection", "exploits", "Buffer Overflow"),
            ("Ransomware", "exploits", "Unpatched Software"),
            ("XSS", "exploits", "Misconfiguration"),
            ("Man-in-the-Middle", "exploits", "Insufficient Encryption"),
            
            # Threats use attack patterns
            ("Phishing", "uses", "Credential Stuffing"),
            ("Malware", "uses", "Code Injection"),
            ("Man-in-the-Middle", "uses", "Session Hijacking"),
            
            # Mitigations protect against threats
            ("Multi-Factor Authentication", "mitigates", "Phishing"),
            ("Multi-Factor Authentication", "mitigates", "Credential Stuffing"),
            ("Patch Management", "mitigates", "Malware"),
            ("Patch Management", "mitigates", "Ransomware"),
            ("Patch Management", "mitigates", "Zero-Day"),
            ("Input Validation", "mitigates", "SQL Injection"),
            ("Input Validation", "mitigates", "XSS"),
            ("Encryption", "mitigates", "Man-in-the-Middle"),
            ("Firewall", "mitigates", "DDoS"),
            ("IDS/IPS", "mitigates", "Malware"),
            ("Security Training", "mitigates", "Phishing"),
            
            # Mitigations address vulnerabilities
            ("Multi-Factor Authentication", "addresses", "Weak Authentication"),
            ("Patch Management", "addresses", "Unpatched Software"),
            ("Input Validation", "addresses", "Buffer Overflow"),
            ("Encryption", "addresses", "Insufficient Encryption"),
        ]
        
        for source, relation, target in relationships:
            self.graph.add_edge(source, target, relation=relation)
    
    def query_related_entities(self, entity: str, max_depth: int = 2) -> Dict[str, List[str]]:
        """
        Query entities related to a given entity up to a certain depth.
        
        Args:
            entity: The entity to query
            max_depth: Maximum depth to traverse
            
        Returns:
            Dictionary of related entities by relation type
        """
        if entity not in self.graph:
            return {}
        
        related = {"direct": [], "indirect": []}
        visited = {entity}
        
        # Direct relationships
        for neighbor in self.graph.neighbors(entity):
            edge_data = self.graph.get_edge_data(entity, neighbor)
            related["direct"].append({
                "entity": neighbor,
                "relation": edge_data.get("relation", "unknown"),
                "attributes": dict(self.graph.nodes[neighbor])
            })
            visited.add(neighbor)
        
        # Indirect relationships
        if max_depth > 1:
            for neighbor in list(visited - {entity}):
                for second_neighbor in self.graph.neighbors(neighbor):
                    if second_neighbor not in visited:
                        edge_data = self.graph.get_edge_data(neighbor, second_neighbor)
                        related["indirect"].append({
                            "entity": second_neighbor,
                            "via": neighbor,
                            "relation": edge_data.get("relation", "unknown"),
                            "attributes": dict(self.graph.nodes[second_neighbor])
                        })
        
        return related
    
    def get_entity_context(self, entity: str) -> Optional[Dict]:
        """Get detailed context about an entity."""
        if entity not in self.graph:
            return None
        
        node_data = dict(self.graph.nodes[entity])
        related = self.query_related_entities(entity, max_depth=2)
        
        return {
            "entity": entity,
            "attributes": node_data,
            "related": related
        }
    
    def find_mitigation_path(self, threat: str) -> List[Dict]:
        """Find mitigation strategies for a given threat."""
        if threat not in self.graph:
            return []
        
        mitigations = []
        
        # Find direct mitigations
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            if node_data.get("type") == "mitigation":
                # Check if there's a path from mitigation to threat
                for neighbor in self.graph.neighbors(node):
                    if neighbor == threat:
                        edge_data = self.graph.get_edge_data(node, neighbor)
                        mitigations.append({
                            "mitigation": node,
                            "path": "direct",
                            "relation": edge_data.get("relation", "unknown"),
                            "effectiveness": node_data.get("effectiveness", "unknown")
                        })
        
        return mitigations
    
    def get_all_entities_by_type(self, entity_type: str) -> List[str]:
        """Get all entities of a specific type."""
        return [
            node for node, data in self.graph.nodes(data=True)
            if data.get("type") == entity_type
        ]
    
    def export_to_dict(self) -> Dict:
        """Export the knowledge graph to a dictionary format."""
        return {
            "nodes": [
                {"id": node, **data}
                for node, data in self.graph.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, **data}
                for u, v, data in self.graph.edges(data=True)
            ]
        }
    
    def get_graph_statistics(self) -> Dict:
        """Get statistics about the knowledge graph."""
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "node_types": {
                node_type: len(self.get_all_entities_by_type(node_type))
                for node_type in ["threat", "vulnerability", "mitigation", "attack_pattern"]
            },
            "avg_degree": sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes()
        }
