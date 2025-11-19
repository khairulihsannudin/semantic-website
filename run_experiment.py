"""
Agentic Graph RAG Experiment Runner
Main script to run and compare Traditional RAG vs Agentic Graph RAG
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, List
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.rag.traditional_rag import TraditionalRAG
from src.rag.agentic_graph_rag import AgenticGraphRAG
from src.evaluation.evaluator import RAGEvaluator
from src.utils.llm_clients import LLMClientFactory
from src.utils.visualization import generate_markdown_report, generate_html_report
from data.sample_data import get_sample_data


class ExperimentRunner:
    """Run experiments comparing different RAG approaches."""
    
    def __init__(self, llm_provider: str = "openai", model: str = "gpt-3.5-turbo"):
        """
        Initialize experiment runner.
        
        Args:
            llm_provider: LLM provider ('openai' or 'anthropic')
            model: Model name to use
        """
        load_dotenv()
        
        self.llm_provider = llm_provider
        self.model = model
        self.evaluator = RAGEvaluator()
        
        # Initialize LLM client
        try:
            self.llm_client = LLMClientFactory.create_client(llm_provider)
            print(f"✓ Initialized {llm_provider} client with model {model}")
        except Exception as e:
            print(f"⚠ Warning: Could not initialize LLM client: {e}")
            print("  Running in demo mode with mock responses")
            self.llm_client = None
        
        # Load sample data
        self.data = get_sample_data()
        print(f"✓ Loaded {len(self.data['documents'])} documents")
        print(f"✓ Loaded {len(self.data['queries'])} test queries")
    
    def run_traditional_rag(self) -> Dict:
        """
        Run Traditional RAG experiment.
        
        Returns:
            Experiment results
        """
        print("\n" + "="*60)
        print("TRADITIONAL RAG EXPERIMENT")
        print("="*60)
        
        # Initialize Traditional RAG
        trad_rag = TraditionalRAG()
        trad_rag.add_documents(self.data['documents'])
        print(f"✓ Initialized Traditional RAG: {trad_rag.get_statistics()}")
        
        # Run queries
        responses = []
        retrieved_docs_list = []
        times = []
        
        for i, query in enumerate(self.data['queries'], 1):
            print(f"\nQuery {i}/{len(self.data['queries'])}: {query}")
            
            start_time = time.time()
            
            if self.llm_client:
                result = trad_rag.generate_response(
                    query,
                    self.llm_client,
                    top_k=3,
                    model=self.model
                )
                response = result['answer']
                retrieved_docs = result['retrieved_documents']
            else:
                # Demo mode
                retrieved_docs = trad_rag.retrieve(query, top_k=3)
                response = f"[Demo Response] Based on retrieved documents: {query}"
            
            elapsed_time = time.time() - start_time
            
            responses.append(response)
            retrieved_docs_list.append(retrieved_docs)
            times.append(elapsed_time)
            
            print(f"  Time: {elapsed_time:.2f}s")
            print(f"  Retrieved: {len(retrieved_docs)} documents")
            print(f"  Response: {response[:100]}...")
        
        # Evaluate
        evaluation = self.evaluator.evaluate_batch(
            self.data['queries'],
            responses,
            self.data['ground_truth'],
            retrieved_docs_list,
            times
        )
        
        return {
            'method': 'Traditional RAG',
            'model': self.model,
            'responses': responses,
            'retrieved_docs': retrieved_docs_list,
            'times': times,
            'evaluation': evaluation
        }
    
    def run_agentic_graph_rag(self) -> Dict:
        """
        Run Agentic Graph RAG experiment.
        
        Returns:
            Experiment results
        """
        print("\n" + "="*60)
        print("AGENTIC GRAPH RAG EXPERIMENT")
        print("="*60)
        
        # Initialize Agentic Graph RAG
        ag_rag = AgenticGraphRAG()
        ag_rag.add_documents(self.data['documents'])
        stats = ag_rag.get_statistics()
        print(f"✓ Initialized Agentic Graph RAG:")
        print(f"  Documents: {stats['num_documents']}")
        print(f"  KG Nodes: {stats['kg_statistics']['num_nodes']}")
        print(f"  KG Edges: {stats['kg_statistics']['num_edges']}")
        
        # Run queries
        responses = []
        retrieved_docs_list = []
        times = []
        kg_entities_list = []
        
        for i, query in enumerate(self.data['queries'], 1):
            print(f"\nQuery {i}/{len(self.data['queries'])}: {query}")
            
            start_time = time.time()
            
            if self.llm_client:
                result = ag_rag.generate_response(
                    query,
                    self.llm_client,
                    top_k=3,
                    model=self.model
                )
                response = result['answer']
                retrieved_docs = result['retrieved_documents']
                num_kg_entities = result['num_kg_entities']
            else:
                # Demo mode
                retrieved_docs = ag_rag.retrieve(query, top_k=3)
                kg_context = ag_rag._expand_with_kg(query, retrieved_docs)
                num_kg_entities = len(kg_context.get('identified_entities', []))
                response = f"[Demo Response] Based on documents and KG: {query}"
            
            elapsed_time = time.time() - start_time
            
            responses.append(response)
            retrieved_docs_list.append(retrieved_docs)
            times.append(elapsed_time)
            kg_entities_list.append(num_kg_entities)
            
            print(f"  Time: {elapsed_time:.2f}s")
            print(f"  Retrieved: {len(retrieved_docs)} documents")
            print(f"  KG Entities: {num_kg_entities}")
            print(f"  Response: {response[:100]}...")
        
        # Evaluate
        evaluation = self.evaluator.evaluate_batch(
            self.data['queries'],
            responses,
            self.data['ground_truth'],
            retrieved_docs_list,
            times
        )
        
        evaluation['avg_kg_entities'] = sum(kg_entities_list) / len(kg_entities_list)
        
        return {
            'method': 'Agentic Graph RAG',
            'model': self.model,
            'responses': responses,
            'retrieved_docs': retrieved_docs_list,
            'times': times,
            'kg_entities': kg_entities_list,
            'evaluation': evaluation
        }
    
    def compare_results(self, trad_results: Dict, ag_results: Dict):
        """
        Compare and display results from both methods.
        
        Args:
            trad_results: Traditional RAG results
            ag_results: Agentic Graph RAG results
        """
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        
        # Create comparison
        comparison = self.evaluator.compare_methods({
            'Traditional RAG': trad_results['evaluation'],
            'Agentic Graph RAG': ag_results['evaluation']
        })
        
        print("\nPerformance Metrics:")
        print("-" * 60)
        
        headers = ["Metric", "Traditional RAG", "Agentic Graph RAG", "Winner"]
        print(f"{headers[0]:<25} {headers[1]:<20} {headers[2]:<20} {headers[3]:<10}")
        print("-" * 60)
        
        # Semantic Similarity
        trad_sim = comparison['metrics']['Traditional RAG']['semantic_similarity']
        ag_sim = comparison['metrics']['Agentic Graph RAG']['semantic_similarity']
        winner_sim = "Agentic" if ag_sim > trad_sim else "Traditional"
        print(f"{'Semantic Similarity':<25} {trad_sim:<20.4f} {ag_sim:<20.4f} {winner_sim:<10}")
        
        # Response Time
        trad_time = comparison['metrics']['Traditional RAG']['response_time']
        ag_time = comparison['metrics']['Agentic Graph RAG']['response_time']
        winner_time = "Traditional" if trad_time < ag_time else "Agentic"
        print(f"{'Avg Response Time (s)':<25} {trad_time:<20.4f} {ag_time:<20.4f} {winner_time:<10}")
        
        # Retrieval Score
        trad_ret = comparison['metrics']['Traditional RAG']['retrieval_score']
        ag_ret = comparison['metrics']['Agentic Graph RAG']['retrieval_score']
        winner_ret = "Agentic" if ag_ret > trad_ret else "Traditional"
        print(f"{'Avg Retrieval Score':<25} {trad_ret:<20.4f} {ag_ret:<20.4f} {winner_ret:<10}")
        
        # Response Length
        trad_len = comparison['metrics']['Traditional RAG']['response_length']
        ag_len = comparison['metrics']['Agentic Graph RAG']['response_length']
        print(f"{'Avg Response Length':<25} {trad_len:<20.1f} {ag_len:<20.1f}")
        
        # KG Enhancement
        if 'avg_kg_entities' in ag_results['evaluation']:
            print(f"{'Avg KG Entities Used':<25} {'-':<20} {ag_results['evaluation']['avg_kg_entities']:<20.1f}")
        
        print("-" * 60)
        
        # Summary
        print("\nSummary:")
        print(f"  • Traditional RAG processed {len(trad_results['queries'])} queries in {sum(trad_results['times']):.2f}s")
        print(f"  • Agentic Graph RAG processed {len(ag_results['queries'])} queries in {sum(ag_results['times']):.2f}s")
        
        if ag_sim > trad_sim:
            improvement = ((ag_sim - trad_sim) / trad_sim) * 100
            print(f"  • Agentic Graph RAG showed {improvement:.1f}% improvement in semantic similarity")
        
        print(f"  • Agentic Graph RAG utilized knowledge graph with average {ag_results['evaluation'].get('avg_kg_entities', 0):.1f} entities per query")
    
    def save_results(self, trad_results: Dict, ag_results: Dict, output_file: str = "results/experiment_results.json"):
        """
        Save experiment results to file.
        
        Args:
            trad_results: Traditional RAG results
            ag_results: Agentic Graph RAG results
            output_file: Output file path
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'llm_provider': self.llm_provider,
            'model': self.model,
            'traditional_rag': {
                'evaluation': trad_results['evaluation'],
                'avg_time': sum(trad_results['times']) / len(trad_results['times'])
            },
            'agentic_graph_rag': {
                'evaluation': ag_results['evaluation'],
                'avg_time': sum(ag_results['times']) / len(ag_results['times'])
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")
        
        # Generate reports
        try:
            md_file = generate_markdown_report(trad_results, ag_results)
            print(f"✓ Markdown report saved to {md_file}")
            
            html_file = generate_html_report(trad_results, ag_results)
            print(f"✓ HTML report saved to {html_file}")
        except Exception as e:
            print(f"⚠ Warning: Could not generate reports: {e}")
    
    def run_full_experiment(self):
        """Run complete experiment comparing both methods."""
        print("="*60)
        print("AGENTIC GRAPH RAG EXPERIMENT")
        print(f"LLM Provider: {self.llm_provider}")
        print(f"Model: {self.model}")
        print("="*60)
        
        # Run experiments
        trad_results = self.run_traditional_rag()
        ag_results = self.run_agentic_graph_rag()
        
        # Compare
        self.compare_results(trad_results, ag_results)
        
        # Save
        self.save_results(trad_results, ag_results)
        
        return trad_results, ag_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run Agentic Graph RAG experiments')
    parser.add_argument(
        '--provider',
        type=str,
        default='openai',
        choices=['openai', 'anthropic'],
        help='LLM provider'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-3.5-turbo',
        help='LLM model name'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run in demo mode without LLM API calls'
    )
    
    args = parser.parse_args()
    
    if args.demo:
        print("Running in DEMO mode (no LLM API calls)\n")
    
    # Run experiment
    runner = ExperimentRunner(llm_provider=args.provider, model=args.model)
    runner.run_full_experiment()
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED")
    print("="*60)


if __name__ == '__main__':
    main()
