"""
Traditional RAG Implementation
Retrieval-Augmented Generation without Knowledge Graphs
"""

from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


class TraditionalRAG:
    """
    Traditional RAG implementation using vector similarity search
    without knowledge graph enhancement.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize Traditional RAG.
        
        Args:
            model_name: Sentence transformer model name
        """
        self.encoder = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
        self.index = None
        self.dimension = None
    
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
                    "similarity": float(1 / (1 + distance))  # Convert distance to similarity
                })
        
        return results
    
    def generate_response(self, query: str, llm_client, top_k: int = 5, model: str = "gpt-3.5-turbo") -> Dict:
        """
        Generate a response using RAG.
        
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
        
        if not retrieved_docs:
            context = "No relevant information found."
        else:
            context = "\n\n".join([
                f"[Document {doc['rank']}]: {doc['document']}"
                for doc in retrieved_docs
            ])
        
        # Create prompt
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""
        
        # Generate response using LLM
        try:
            if hasattr(llm_client, 'chat') and hasattr(llm_client.chat, 'completions'):
                # OpenAI client
                response = llm_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful cybersecurity expert."},
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
            "num_retrieved": len(retrieved_docs),
            "method": "Traditional RAG"
        }
    
    def get_statistics(self) -> Dict:
        """Get statistics about the RAG system."""
        return {
            "num_documents": len(self.documents),
            "embedding_dimension": self.dimension,
            "index_size": self.index.ntotal if self.index else 0
        }
