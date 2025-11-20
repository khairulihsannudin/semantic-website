"""
Agentic Graph RAG Implementation
RAG enhanced with Knowledge Graphs and agentic reasoning
"""

from typing import List, Dict, Optional, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from ..knowledge_graph.cskg import CybersecurityKnowledgeGraph


class AgenticGraphRAG:
    """
    Agentic Graph RAG implementation that combines vector similarity search
    with knowledge graph reasoning for enhanced retrieval and generation.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize Agentic Graph RAG.
        
        Args:
            model_name: Sentence transformer model name
        """
        self.encoder = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
        self.index = None
        self.dimension = None
        self.kg = CybersecurityKnowledgeGraph()
    
    def add_documents(self, documents: List[str]):
        """
        Add documents to the RAG system.
        
        Args:
            documents: List of document strings
        """
        self.documents.extend(documents)
        
        # Encode documents
        doc_embeddings = self.encoder.encode(documents, show_progress_bar=False)
        
        if self.embeddings is None:
            self.embeddings = doc_embeddings
            self.dimension = doc_embeddings.shape[1]
            # Initialize FAISS index
            self.index = faiss.IndexFlatL2(self.dimension)
        else:
            self.embeddings = np.vstack([self.embeddings, doc_embeddings])
        
        # Add to index
        self.index.add(doc_embeddings.astype('float32'))
    
    def _extract_entities(self, text: str) -> List[str]:
        """
        Extract cybersecurity entities from text using the knowledge graph.
        
        Args:
            text: Input text
            
        Returns:
            List of identified entities
        """
        entities = []
        text_lower = text.lower()
        
        # Check all nodes in the knowledge graph
        for node in self.kg.graph.nodes():
            if node.lower() in text_lower:
                entities.append(node)
        
        return entities
    
    def _expand_with_kg(self, query: str, retrieved_docs: List[Dict]) -> Dict:
        """
        Expand retrieved information with knowledge graph context.
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents
            
        Returns:
            Enhanced context with KG information
        """
        # Extract entities from query and retrieved documents
        query_entities = self._extract_entities(query)
        
        doc_entities = set()
        for doc in retrieved_docs:
            doc_entities.update(self._extract_entities(doc["document"]))
        
        all_entities = list(set(query_entities + list(doc_entities)))
        
        # Get KG context for identified entities
        kg_context = {}
        for entity in all_entities:
            context = self.kg.get_entity_context(entity)
            if context:
                kg_context[entity] = context
        
        # Find relevant paths and relationships
        enhanced_info = {
            "identified_entities": all_entities,
            "entity_contexts": kg_context,
            "mitigations": [],
            "related_threats": [],
            "vulnerabilities": []
        }
        
        # Extract specific information based on query intent
        for entity in all_entities:
            node_data = self.kg.graph.nodes.get(entity, {})
            entity_type = node_data.get("type")
            
            if entity_type == "threat":
                mitigations = self.kg.find_mitigation_path(entity)
                enhanced_info["mitigations"].extend(mitigations)
                enhanced_info["related_threats"].append(entity)
            elif entity_type == "vulnerability":
                enhanced_info["vulnerabilities"].append(entity)
        
        return enhanced_info
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with scores
        """
        if self.index is None or len(self.documents) == 0:
            return []
        
        # Encode query
        query_embedding = self.encoder.encode([query], show_progress_bar=False)
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), min(top_k, len(self.documents)))
        
        results = []
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.documents):
                results.append({
                    "rank": i + 1,
                    "document": self.documents[idx],
                    "score": float(distance),
                    "similarity": float(1 / (1 + distance))
                })
        
        return results
    
    def _create_enhanced_prompt(self, query: str, retrieved_docs: List[Dict], kg_context: Dict) -> str:
        """
        Create an enhanced prompt with both retrieved documents and KG context.
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents
            kg_context: Knowledge graph context
            
        Returns:
            Enhanced prompt string
        """
        prompt_parts = ["Based on the following information, answer the question:\n"]
        
        # Add retrieved documents
        if retrieved_docs:
            prompt_parts.append("Retrieved Context:")
            for doc in retrieved_docs:
                prompt_parts.append(f"[Document {doc['rank']}]: {doc['document']}")
            prompt_parts.append("")
        
        # Add knowledge graph context
        if kg_context.get("identified_entities"):
            prompt_parts.append("Knowledge Graph Context:")
            
            for entity in kg_context["identified_entities"]:
                entity_context = kg_context["entity_contexts"].get(entity)
                if entity_context:
                    attrs = entity_context["attributes"]
                    prompt_parts.append(f"- {entity} ({attrs.get('type', 'unknown')}): {attrs.get('description', 'No description')}")
                    
                    # Add related entities
                    if entity_context["related"]["direct"]:
                        related_strs = []
                        for rel in entity_context["related"]["direct"][:3]:  # Limit to top 3
                            related_strs.append(f"{rel['relation']} {rel['entity']}")
                        prompt_parts.append(f"  Related: {', '.join(related_strs)}")
            
            prompt_parts.append("")
            
            # Add mitigations if found
            if kg_context.get("mitigations"):
                prompt_parts.append("Recommended Mitigations:")
                for mitigation in kg_context["mitigations"][:3]:  # Top 3
                    prompt_parts.append(f"- {mitigation['mitigation']} (effectiveness: {mitigation['effectiveness']})")
                prompt_parts.append("")
        
        # Add the query
        prompt_parts.append(f"Question: {query}\n")
        prompt_parts.append("Answer (provide a comprehensive response using both retrieved documents and knowledge graph information):")
        
        return "\n".join(prompt_parts)
    
    def generate_response(self, query: str, llm_client, top_k: int = 5, model: str = "gpt-3.5-turbo") -> Dict:
        """
        Generate a response using Agentic Graph RAG.
        
        Args:
            query: User query
            llm_client: LLM client (OpenAI or Anthropic)
            top_k: Number of documents to retrieve
            model: LLM model name
            
        Returns:
            Dictionary with response and metadata
        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query, top_k=top_k)
        
        # Enhance with knowledge graph
        kg_context = self._expand_with_kg(query, retrieved_docs)
        
        # Create enhanced prompt
        prompt = self._create_enhanced_prompt(query, retrieved_docs, kg_context)
        
        # Generate response using LLM
        try:
            if hasattr(llm_client, 'chat') and hasattr(llm_client.chat, 'completions'):
                # OpenAI client
                response = llm_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful cybersecurity expert with access to a knowledge graph. Provide detailed, accurate answers using the provided context and knowledge."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                answer = response.choices[0].message.content
            else:
                # Anthropic client
                response = llm_client.messages.create(
                    model=model,
                    max_tokens=500,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                answer = response.content[0].text
        except Exception as e:
            answer = f"Error generating response: {str(e)}"
        
        return {
            "query": query,
            "answer": answer,
            "retrieved_documents": retrieved_docs,
            "kg_context": kg_context,
            "num_retrieved": len(retrieved_docs),
            "num_kg_entities": len(kg_context.get("identified_entities", [])),
            "method": "Agentic Graph RAG"
        }
    
    def get_statistics(self) -> Dict:
        """Get statistics about the Agentic Graph RAG system."""
        return {
            "num_documents": len(self.documents),
            "embedding_dimension": self.dimension,
            "index_size": self.index.ntotal if self.index else 0,
            "kg_statistics": self.kg.get_graph_statistics()
        }
