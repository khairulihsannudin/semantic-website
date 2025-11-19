# Usage Guide

## Quick Start

### 1. Basic Demo (No API Keys Required)

Run the quick demo to see the Knowledge Graph and understand the approach:

```bash
python demo.py
```

This will show:
- Cybersecurity Knowledge Graph structure
- Example entity queries
- Comparison between Traditional RAG and Agentic Graph RAG
- Expected performance metrics

### 2. Run Experiment in Demo Mode

Test the full experiment pipeline without LLM API calls:

```bash
python run_experiment.py --demo
```

This will:
- Initialize both RAG systems
- Process test queries
- Perform mock retrievals
- Generate comparison metrics
- Save results to `results/` folder

### 3. Run with OpenAI

Set up your API key and run with OpenAI:

```bash
# Set environment variable
export OPENAI_API_KEY="your-api-key-here"

# Run with GPT-3.5-turbo (faster, cheaper)
python run_experiment.py --provider openai --model gpt-3.5-turbo

# Or run with GPT-4 (better quality, slower)
python run_experiment.py --provider openai --model gpt-4
```

### 4. Run with Anthropic Claude

```bash
# Set environment variable
export ANTHROPIC_API_KEY="your-api-key-here"

# Run with Claude
python run_experiment.py --provider anthropic --model claude-3-haiku-20240307
```

### 5. Multi-LLM Comparison

Test multiple models at once:

```bash
python run_multi_llm.py --models openai:gpt-3.5-turbo openai:gpt-4 anthropic:claude-3-haiku-20240307
```

## Command Line Options

### run_experiment.py

```
--provider    LLM provider (openai or anthropic)
--model       Model name (e.g., gpt-3.5-turbo, claude-3-sonnet-20240229)
--demo        Run without LLM API calls (mock responses)
```

### run_multi_llm.py

```
--models      Space-separated list of models (format: provider:model)
--demo        Run in demo mode (will skip actual LLM calls)
```

## Understanding the Output

### During Experiment

The experiment will show:
1. **Initialization**: Loading models and data
2. **Traditional RAG**: Processing queries with basic retrieval
3. **Agentic Graph RAG**: Processing with KG enhancement
4. **Comparison**: Side-by-side metrics

### Results Files

After running, check the `results/` folder:

- `experiment_results.json`: Raw data and metrics
- `report.md`: Human-readable Markdown report
- `report.html`: HTML report with visualizations (open in browser)

### Key Metrics

**Semantic Similarity** (0-1, higher is better)
- Measures how close the generated answer is to ground truth
- Agentic Graph RAG typically scores higher

**Response Time** (seconds, lower is better)
- Time to generate each response
- Traditional RAG is faster

**Retrieval Score** (0-1, higher is better)
- Quality of retrieved documents
- Higher score = better document matching

**Response Length** (word count)
- Comprehensiveness of answers
- Agentic Graph RAG tends to be more detailed

**KG Entities Used** (count)
- Number of knowledge graph entities utilized
- Only available for Agentic Graph RAG

## Test Queries

The system is evaluated on 10 cybersecurity queries:

1. What is phishing and how can I protect against it?
2. How does ransomware work and what are the best defenses?
3. What is SQL injection and how can it be prevented?
4. Why is multi-factor authentication important?
5. What are the best practices for preventing DDoS attacks?
6. How can organizations protect against zero-day vulnerabilities?
7. What is a man-in-the-middle attack?
8. What are the key components of effective patch management?
9. How does encryption help protect data?
10. What mitigations are effective against malware?

## Customization

### Add Your Own Documents

Edit `data/sample_data.py` to add more cybersecurity documents:

```python
CYBERSECURITY_DOCUMENTS = [
    "Your new document here...",
    # ... more documents
]
```

### Add Custom Queries

Add test queries in `data/sample_data.py`:

```python
TEST_QUERIES = [
    "Your custom query?",
    # ... more queries
]
```

### Extend the Knowledge Graph

Modify `src/knowledge_graph/cskg.py` to add entities:

```python
def _initialize_cskg(self):
    threats = [
        ("New Threat", {"type": "threat", "severity": "high", ...}),
        # ... more threats
    ]
    # Add relationships
    relationships = [
        ("Entity1", "relation_type", "Entity2"),
        # ... more relationships
    ]
```

## Troubleshooting

### Import Errors

If you get import errors, ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

### Slow Performance

The first run may be slow due to:
- Downloading sentence transformer models
- Initializing embeddings

Subsequent runs will be faster.

### API Rate Limits

If you hit rate limits:
- Use demo mode: `--demo`
- Reduce queries in `data/sample_data.py`
- Use a lower-tier model (e.g., gpt-3.5-turbo)

### Memory Issues

If you encounter memory issues:
- Close other applications
- Use a smaller embedding model
- Process fewer documents

## Advanced Usage

### Programmatic Usage

Use the classes directly in your code:

```python
from src.knowledge_graph.cskg import CybersecurityKnowledgeGraph
from src.rag.agentic_graph_rag import AgenticGraphRAG
from src.utils.llm_clients import LLMClientFactory

# Initialize
kg = CybersecurityKnowledgeGraph()
rag = AgenticGraphRAG()

# Add documents
rag.add_documents(your_documents)

# Query
client = LLMClientFactory.create_client('openai')
result = rag.generate_response("Your query", client, model='gpt-3.5-turbo')
print(result['answer'])
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_cskg.py

# Run with verbose output
pytest tests/ -v
```

## Best Practices

1. **Start with demo mode** to understand the system
2. **Use GPT-3.5-turbo** for initial testing (faster, cheaper)
3. **Review HTML reports** for detailed insights
4. **Test multiple models** to find the best fit
5. **Customize the KG** for your specific domain

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check the README.md for detailed information
- Review the code documentation in source files
