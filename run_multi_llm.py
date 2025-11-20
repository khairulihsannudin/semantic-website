"""
Multi-LLM Experiment Runner
Tests the RAG systems with multiple LLM models for comparison
"""

import os
import sys
import argparse
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from run_experiment import ExperimentRunner


def run_multi_llm_experiments(providers_models: list):
    """
    Run experiments with multiple LLM models.
    
    Args:
        providers_models: List of (provider, model) tuples
    """
    print("="*60)
    print("MULTI-LLM EXPERIMENT")
    print("="*60)
    print(f"\nTesting {len(providers_models)} different LLM configurations\n")
    
    all_results = []
    
    for i, (provider, model) in enumerate(providers_models, 1):
        print(f"\n{'='*60}")
        print(f"Experiment {i}/{len(providers_models)}: {provider} - {model}")
        print(f"{'='*60}\n")
        
        try:
            runner = ExperimentRunner(llm_provider=provider, model=model)
            trad_results, ag_results = runner.run_full_experiment()
            
            all_results.append({
                'provider': provider,
                'model': model,
                'traditional_rag': trad_results['evaluation']['aggregated'],
                'agentic_graph_rag': ag_results['evaluation']['aggregated']
            })
            
        except Exception as e:
            print(f"✗ Error running experiment with {provider}/{model}: {e}")
            continue
    
    # Summary comparison
    if all_results:
        print("\n" + "="*60)
        print("MULTI-LLM COMPARISON SUMMARY")
        print("="*60)
        
        print("\n{:<25} {:<20} {:<15} {:<15}".format(
            "Model", "Method", "Sem. Similarity", "Avg Time (s)"
        ))
        print("-"*75)
        
        for result in all_results:
            model_name = f"{result['provider']}/{result['model']}"
            
            # Traditional RAG
            trad = result['traditional_rag']
            print("{:<25} {:<20} {:<15.4f} {:<15.4f}".format(
                model_name[:24], "Traditional RAG",
                trad.get('avg_semantic_similarity', 0),
                trad.get('avg_response_time', 0)
            ))
            
            # Agentic Graph RAG
            ag = result['agentic_graph_rag']
            print("{:<25} {:<20} {:<15.4f} {:<15.4f}".format(
                "", "Agentic Graph RAG",
                ag.get('avg_semantic_similarity', 0),
                ag.get('avg_response_time', 0)
            ))
            print("-"*75)
        
        # Find best performing
        best_trad = max(all_results, 
                       key=lambda x: x['traditional_rag'].get('avg_semantic_similarity', 0))
        best_ag = max(all_results,
                     key=lambda x: x['agentic_graph_rag'].get('avg_semantic_similarity', 0))
        
        print("\nBest Performers:")
        print(f"  Traditional RAG: {best_trad['provider']}/{best_trad['model']}")
        print(f"    Semantic Similarity: {best_trad['traditional_rag'].get('avg_semantic_similarity', 0):.4f}")
        print(f"  Agentic Graph RAG: {best_ag['provider']}/{best_ag['model']}")
        print(f"    Semantic Similarity: {best_ag['agentic_graph_rag'].get('avg_semantic_similarity', 0):.4f}")
    
    print("\n" + "="*60)
    print("MULTI-LLM EXPERIMENT COMPLETED")
    print("="*60)
    
    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run multi-LLM RAG experiments')
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        help='List of models to test (format: provider:model)'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run in demo mode without LLM API calls'
    )
    
    args = parser.parse_args()
    
    load_dotenv()
    
    # Default models to test
    default_models = [
        ('openai', 'gpt-3.5-turbo'),
        ('openai', 'gpt-4'),
    ]
    
    if args.models:
        # Parse custom models
        providers_models = []
        for model_str in args.models:
            if ':' in model_str:
                provider, model = model_str.split(':', 1)
                providers_models.append((provider, model))
            else:
                print(f"Warning: Invalid format '{model_str}', expected 'provider:model'")
        
        if not providers_models:
            providers_models = default_models
    else:
        providers_models = default_models
    
    if args.demo:
        print("Running in DEMO mode (no LLM API calls)")
        print("Multi-LLM comparison requires actual LLM API access")
        return
    
    # Run experiments
    run_multi_llm_experiments(providers_models)


if __name__ == '__main__':
    main()
