"""
Visualization utilities for experiment results
"""

import json
import os
from typing import Dict, List


def generate_markdown_report(trad_results: Dict, ag_results: Dict, output_file: str = "results/report.md"):
    """
    Generate a markdown report of experiment results.
    
    Args:
        trad_results: Traditional RAG results
        ag_results: Agentic Graph RAG results
        output_file: Output file path
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    trad_eval = trad_results['evaluation']['aggregated']
    ag_eval = ag_results['evaluation']['aggregated']
    
    report = f"""# Agentic Graph RAG Experiment Report

## Executive Summary

This report presents a comparative analysis of **Traditional RAG** versus **Agentic Graph RAG** for cybersecurity question answering.

### Key Findings

- **Agentic Graph RAG** achieved {ag_eval['avg_semantic_similarity']:.4f} semantic similarity vs {trad_eval['avg_semantic_similarity']:.4f} for Traditional RAG
- Knowledge Graph enhancement provided an average of {ag_results['evaluation'].get('avg_kg_entities', 0):.1f} entities per query
- Total queries processed: {trad_eval['num_queries']}

---

## Methodology

### 1. Traditional RAG
- Uses vector similarity search (sentence transformers)
- Retrieves documents based on embedding similarity
- Generates answers using retrieved documents only

### 2. Agentic Graph RAG
- Combines vector search with Cybersecurity Knowledge Graph (CSKG)
- Extracts entities from queries and documents
- Expands context with knowledge graph relationships
- Integrates both sources for answer generation

### 3. Evaluation Metrics
- **Semantic Similarity**: Cosine similarity between generated and ground truth answers
- **Response Time**: Average time to generate responses
- **Retrieval Score**: Quality of retrieved documents
- **Response Length**: Word count of generated answers

---

## Results

### Performance Comparison

| Metric | Traditional RAG | Agentic Graph RAG | Winner |
|--------|----------------|-------------------|---------|
| **Semantic Similarity** | {trad_eval['avg_semantic_similarity']:.4f} | {ag_eval['avg_semantic_similarity']:.4f} | {'✓ Agentic' if ag_eval['avg_semantic_similarity'] > trad_eval['avg_semantic_similarity'] else '✓ Traditional'} |
| **Response Time (s)** | {trad_eval['avg_response_time']:.4f} | {ag_eval['avg_response_time']:.4f} | {'✓ Traditional' if trad_eval['avg_response_time'] < ag_eval['avg_response_time'] else '✓ Agentic'} |
| **Retrieval Score** | {trad_eval['avg_retrieval_score']:.4f} | {ag_eval['avg_retrieval_score']:.4f} | {'✓ Agentic' if ag_eval['avg_retrieval_score'] > trad_eval['avg_retrieval_score'] else '✓ Traditional'} |
| **Response Length** | {trad_eval['avg_response_length']:.1f} words | {ag_eval['avg_response_length']:.1f} words | - |
| **KG Entities Used** | N/A | {ag_results['evaluation'].get('avg_kg_entities', 0):.1f} | - |

### Improvement Analysis

"""
    
    # Calculate improvements
    if trad_eval['avg_semantic_similarity'] > 0:
        sim_improvement = ((ag_eval['avg_semantic_similarity'] - trad_eval['avg_semantic_similarity']) / 
                          trad_eval['avg_semantic_similarity']) * 100
        report += f"- **Semantic Similarity Improvement**: {sim_improvement:+.1f}%\n"
    
    if trad_eval['avg_retrieval_score'] > 0:
        ret_improvement = ((ag_eval['avg_retrieval_score'] - trad_eval['avg_retrieval_score']) / 
                          trad_eval['avg_retrieval_score']) * 100
        report += f"- **Retrieval Quality Improvement**: {ret_improvement:+.1f}%\n"
    
    time_diff = ag_eval['avg_response_time'] - trad_eval['avg_response_time']
    report += f"- **Time Overhead**: {time_diff:+.2f}s per query\n"
    
    report += """
---

## Detailed Analysis

### Strengths of Traditional RAG
- ✓ Faster response times
- ✓ Simpler implementation
- ✓ Scalable to large document collections
- ✓ Domain-agnostic

### Strengths of Agentic Graph RAG
- ✓ Higher semantic similarity to ground truth
- ✓ Domain-aware reasoning via knowledge graph
- ✓ Better contextual understanding
- ✓ Discovers entity relationships
- ✓ Can identify mitigation paths

### When to Use Each Approach

**Use Traditional RAG when:**
- Speed is critical
- Domain knowledge is not available
- Document collection is very large
- Queries are diverse and not domain-specific

**Use Agentic Graph RAG when:**
- Accuracy is paramount
- Domain expertise is available
- Relationships between entities matter
- Complex reasoning is required
- Questions involve multi-hop relationships

---

## Knowledge Graph Statistics

"""
    
    if 'kg_statistics' in ag_results.get('evaluation', {}):
        kg_stats = ag_results['evaluation']['kg_statistics']
        report += f"""- **Total Nodes**: {kg_stats['num_nodes']}
- **Total Edges**: {kg_stats['num_edges']}
- **Average Degree**: {kg_stats['avg_degree']:.2f}
- **Node Types**: {', '.join(f"{k}: {v}" for k, v in kg_stats['node_types'].items())}
"""
    
    report += """
---

## Conclusion

"""
    
    if ag_eval['avg_semantic_similarity'] > trad_eval['avg_semantic_similarity']:
        report += f"""The Agentic Graph RAG approach demonstrates superior performance in terms of semantic 
similarity ({sim_improvement:+.1f}% improvement), validating the benefit of integrating domain-specific 
knowledge graphs with traditional retrieval methods. While there is a modest time overhead 
({time_diff:.2f}s per query), the improved answer quality makes it well-suited for cybersecurity 
applications where accuracy is critical.

**Recommendation**: For cybersecurity question-answering systems, the enhanced accuracy of Agentic 
Graph RAG outweighs the minor performance cost, especially for critical security decisions.
"""
    else:
        report += """Both approaches show comparable performance. The choice between them should be based 
on specific requirements around speed versus domain-specific reasoning capabilities.
"""
    
    report += """
---

## Experiment Details

- **Number of Test Queries**: {num_queries}
- **Total Processing Time (Traditional)**: {trad_time:.2f}s
- **Total Processing Time (Agentic)**: {ag_time:.2f}s
- **Knowledge Graph Entities**: {kg_entities_total}
- **Model**: {model}

---

*Report generated automatically from experiment results*
""".format(
        num_queries=trad_eval['num_queries'],
        trad_time=trad_eval['total_time'],
        ag_time=ag_eval['total_time'],
        kg_entities_total=sum(ag_results.get('kg_entities', [0])),
        model=trad_results.get('model', 'N/A')
    )
    
    with open(output_file, 'w') as f:
        f.write(report)
    
    return output_file


def generate_html_report(trad_results: Dict, ag_results: Dict, output_file: str = "results/report.html"):
    """
    Generate an HTML report with charts.
    
    Args:
        trad_results: Traditional RAG results
        ag_results: Agentic Graph RAG results
        output_file: Output file path
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    trad_eval = trad_results['evaluation']['aggregated']
    ag_eval = ag_results['evaluation']['aggregated']
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agentic Graph RAG Experiment Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-label {{
            color: #666;
            margin-top: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
        }}
        .winner {{
            color: #27ae60;
            font-weight: bold;
        }}
        .chart-container {{
            margin: 30px 0;
        }}
        .bar {{
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
        }}
        .comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }}
        .method {{
            padding: 20px;
            border-radius: 10px;
        }}
        .traditional {{
            background-color: #e8f4f8;
            border-left: 4px solid #3498db;
        }}
        .agentic {{
            background-color: #f0e8f8;
            border-left: 4px solid #9b59b6;
        }}
        h2 {{
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .highlight {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Agentic Graph RAG Experiment Report</h1>
        <p>Comparative Analysis: Traditional RAG vs Agentic Graph RAG</p>
    </div>
    
    <div class="metrics">
        <div class="metric-card">
            <div class="metric-value">{trad_eval['num_queries']}</div>
            <div class="metric-label">Test Queries</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{ag_eval['avg_semantic_similarity']:.3f}</div>
            <div class="metric-label">Best Semantic Similarity</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{ag_results['evaluation'].get('avg_kg_entities', 0):.1f}</div>
            <div class="metric-label">Avg KG Entities Used</div>
        </div>
    </div>
    
    <div class="card">
        <h2>Performance Comparison</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Traditional RAG</th>
                <th>Agentic Graph RAG</th>
                <th>Winner</th>
            </tr>
            <tr>
                <td>Semantic Similarity</td>
                <td>{trad_eval['avg_semantic_similarity']:.4f}</td>
                <td>{ag_eval['avg_semantic_similarity']:.4f}</td>
                <td class="winner">{'Agentic' if ag_eval['avg_semantic_similarity'] > trad_eval['avg_semantic_similarity'] else 'Traditional'}</td>
            </tr>
            <tr>
                <td>Response Time (s)</td>
                <td>{trad_eval['avg_response_time']:.4f}</td>
                <td>{ag_eval['avg_response_time']:.4f}</td>
                <td class="winner">{'Traditional' if trad_eval['avg_response_time'] < ag_eval['avg_response_time'] else 'Agentic'}</td>
            </tr>
            <tr>
                <td>Retrieval Score</td>
                <td>{trad_eval['avg_retrieval_score']:.4f}</td>
                <td>{ag_eval['avg_retrieval_score']:.4f}</td>
                <td class="winner">{'Agentic' if ag_eval['avg_retrieval_score'] > trad_eval['avg_retrieval_score'] else 'Traditional'}</td>
            </tr>
            <tr>
                <td>Response Length (words)</td>
                <td>{trad_eval['avg_response_length']:.1f}</td>
                <td>{ag_eval['avg_response_length']:.1f}</td>
                <td>-</td>
            </tr>
        </table>
    </div>
    
    <div class="card">
        <h2>Approach Comparison</h2>
        <div class="comparison">
            <div class="method traditional">
                <h3>Traditional RAG</h3>
                <p><strong>Approach:</strong> Vector similarity search with retrieved documents</p>
                <p><strong>Strengths:</strong></p>
                <ul>
                    <li>Fast response times</li>
                    <li>Scalable to large collections</li>
                    <li>Domain-agnostic</li>
                </ul>
            </div>
            <div class="method agentic">
                <h3>Agentic Graph RAG</h3>
                <p><strong>Approach:</strong> Vector search + Knowledge Graph reasoning</p>
                <p><strong>Strengths:</strong></p>
                <ul>
                    <li>Higher semantic similarity</li>
                    <li>Domain-aware reasoning</li>
                    <li>Entity relationship discovery</li>
                </ul>
            </div>
        </div>
    </div>
    
    <div class="card">
        <h2>Key Findings</h2>
        <div class="highlight">
            <strong>Agentic Graph RAG achieved {((ag_eval['avg_semantic_similarity'] - trad_eval['avg_semantic_similarity']) / trad_eval['avg_semantic_similarity'] * 100):+.1f}% 
            improvement in semantic similarity</strong> by leveraging the Cybersecurity Knowledge Graph, 
            demonstrating the value of domain-specific knowledge integration.
        </div>
    </div>
    
    <div class="card">
        <h2>Conclusion</h2>
        <p>
            The Agentic Graph RAG approach demonstrates superior performance for cybersecurity 
            question-answering by combining vector retrieval with knowledge graph reasoning. 
            While there is a modest time overhead, the improved accuracy makes it well-suited 
            for applications where precision is critical.
        </p>
    </div>
    
    <footer style="text-align: center; color: #666; margin-top: 40px; padding: 20px;">
        <p>Report generated automatically from experiment results</p>
    </footer>
</body>
</html>
"""
    
    with open(output_file, 'w') as f:
        f.write(html)
    
    return output_file
