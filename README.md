# Agentic Graph RAG for Cybersecurity

This repository implements and compares **Traditional RAG** (Retrieval-Augmented Generation) with **Agentic Graph RAG** for cybersecurity knowledge reasoning. The implementation is based on recent research combining Knowledge Graphs with RAG techniques to enhance intelligent cybersecurity reasoning.

## Overview

The project provides:
- **Traditional RAG**: Vector-based retrieval without knowledge graphs
- **Agentic Graph RAG**: Enhanced retrieval using Cybersecurity Knowledge Graphs (CSKG)
- **Performance Evaluation**: Speed, accuracy, and relevance metrics
- **Multi-LLM Support**: Compatible with OpenAI and Anthropic models

## Features

### 1. Cybersecurity Knowledge Graph (CSKG)
- Pre-built graph with cybersecurity entities (threats, vulnerabilities, mitigations, attack patterns)
- Entity relationships and attributes
- Graph-based reasoning and path finding

### 2. Traditional RAG
- Vector similarity search using sentence transformers
- FAISS indexing for efficient retrieval
- Basic context augmentation

### 3. Agentic Graph RAG
- Combines vector search with knowledge graph reasoning
- Entity extraction and context expansion
- Graph-aware prompt enhancement
- Mitigation path discovery

### 4. Evaluation Framework
- Semantic similarity metrics
- Retrieval performance metrics
- Response time measurement
- Comparative analysis

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/khairulihsannudin/semantic-website.git
cd semantic-website
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API keys (optional, for LLM integration):
```bash
cp .env.example .env
# Edit .env and add your API keys
```

## Usage

### Running the Experiment

#### Demo Mode (No API Keys Required)
```bash
python run_experiment.py --demo
```

#### With OpenAI
```bash
export OPENAI_API_KEY="your-key-here"
python run_experiment.py --provider openai --model gpt-3.5-turbo
```

#### With Anthropic Claude
```bash
export ANTHROPIC_API_KEY="your-key-here"
python run_experiment.py --provider anthropic --model claude-3-haiku-20240307
```

### Command Line Options

- `--provider`: LLM provider (`openai` or `anthropic`, default: `openai`)
- `--model`: Model name (e.g., `gpt-4`, `claude-3-sonnet-20240229`)
- `--demo`: Run without LLM API calls (uses mock responses)

## Project Structure

```
semantic-website/
├── src/
│   ├── knowledge_graph/    # CSKG implementation
│   │   └── cskg.py
│   ├── rag/                # RAG implementations
│   │   ├── traditional_rag.py
│   │   └── agentic_graph_rag.py
│   ├── evaluation/         # Evaluation framework
│   │   └── evaluator.py
│   └── utils/              # Utility modules
│       └── llm_clients.py
├── data/
│   └── sample_data.py      # Test data and queries
├── tests/                  # Unit tests
├── run_experiment.py       # Main experiment runner
├── requirements.txt        # Dependencies
└── README.md
```

## Experiment Details

### Test Queries
The system is evaluated on 10 cybersecurity queries covering:
- Phishing attacks
- Ransomware
- SQL injection
- Multi-factor authentication
- DDoS attacks
- Zero-day vulnerabilities
- Man-in-the-middle attacks
- Patch management
- Encryption
- Malware mitigations

### Knowledge Graph Structure
- **Nodes**: 25+ cybersecurity entities
- **Node Types**: threats, vulnerabilities, mitigations, attack_patterns
- **Edges**: Relationships (exploits, mitigates, uses, addresses)

### Evaluation Metrics
1. **Semantic Similarity**: Cosine similarity between generated response and ground truth
2. **Response Time**: Average time to generate responses
3. **Retrieval Score**: Quality of retrieved documents
4. **Response Length**: Comprehensiveness of answers

## Results

The experiment compares:
- **Accuracy**: Agentic Graph RAG typically shows improved semantic similarity due to knowledge graph context
- **Speed**: Traditional RAG is faster; Agentic Graph RAG trades some speed for accuracy
- **Relevance**: Agentic Graph RAG provides more contextually relevant answers with entity relationships

Example output:
```
Performance Metrics:
----------------------------------------------------------------
Metric                    Traditional RAG      Agentic Graph RAG    Winner    
----------------------------------------------------------------
Semantic Similarity       0.7234              0.7891              Agentic   
Avg Response Time (s)     1.23                1.67                Traditional
Avg Retrieval Score       0.6543              0.6789              Agentic   
Avg Response Length       87.3                124.5               
Avg KG Entities Used      -                   3.2                 
----------------------------------------------------------------
```

## Supported LLM Models

### OpenAI
- gpt-4
- gpt-4-turbo-preview
- gpt-3.5-turbo
- gpt-3.5-turbo-16k

### Anthropic
- claude-3-opus-20240229
- claude-3-sonnet-20240229
- claude-3-haiku-20240307
- claude-2.1

## Testing

Run unit tests:
```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

See LICENSE file for details.

## Citation

If you use this code for research, please cite:
```
@misc{agentic-graph-rag-cyber,
  title={Agentic Graph RAG for Cybersecurity},
  author={Semantic Website Project},
  year={2025},
  url={https://github.com/khairulihsannudin/semantic-website}
}
```

## Acknowledgments

This implementation is inspired by recent research in:
- Retrieval-Augmented Generation (RAG)
- Knowledge Graph-enhanced AI systems
- Cybersecurity knowledge representation
- Agentic AI reasoning