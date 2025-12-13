# Evaluation Guide

This guide explains how to use the evaluation system for the multi-agent RAG system.

## Overview

The evaluation system measures the performance of the multi-agent RAG system across three key metrics:

1. **Speed Metrics**: Execution time for the entire workflow and individual nodes
2. **Accuracy Metrics**: How well the generated answers match ground truth
3. **Relevance Metrics**: How relevant the answers and contexts are to the questions

## Quick Start

### Running an Evaluation

```bash
# Run evaluation with the sample dataset
uv run -m src.evaluation.run_evaluation data/evaluation/sample_eval_dataset.json

# Run with custom configuration
uv run -m src.evaluation.run_evaluation data/evaluation/sample_eval_dataset.json \
    --config config/evaluation_config.yaml

# Run with custom output directory
uv run -m src.evaluation.run_evaluation data/evaluation/sample_eval_dataset.json \
    --output ./my_reports

# Run with verbose logging
uv run -m src.evaluation.run_evaluation data/evaluation/sample_eval_dataset.json --verbose
```

### Programmatic Usage

```python
import asyncio
from src.evaluation.dataset import EvaluationDataset
from src.evaluation.evaluator import Evaluator
from src.evaluation.reporter import EvaluationReporter

async def run_evaluation():
    # Load dataset
    dataset = EvaluationDataset.from_json("data/evaluation/sample_eval_dataset.json")
    
    # Create evaluator
    evaluator = Evaluator()
    
    # Run evaluation
    results = await evaluator.evaluate_dataset(dataset)
    
    # Generate summary
    summary = evaluator.generate_summary(dataset)
    
    # Generate reports
    reporter = EvaluationReporter(
        results=results,
        summary=summary,
        output_dir="./evaluation_reports"
    )
    
    # Generate all report formats
    report_paths = reporter.generate_all_reports()
    
    return summary

# Run
summary = asyncio.run(run_evaluation())
print(f"Mean Semantic Similarity: {summary.aggregate_accuracy['mean_semantic_similarity']}")
```

## Metrics Explained

### Speed Metrics

| Metric | Description |
|--------|-------------|
| `total_execution_time_seconds` | End-to-end query processing time |
| `per_node_times` | Time spent in each agent node |
| `retrieval_times` | Separate timing for vector/cypher/MCP RDF operations |
| `latency_breakdown` | Detailed timing breakdown by pipeline stage |

### Accuracy Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `exact_match` | Binary score for exact text match | 0 or 1 |
| `semantic_similarity` | Cosine similarity of embeddings | 0 to 1 |
| `rouge_scores` | ROUGE-1, ROUGE-2, ROUGE-L F1 scores | 0 to 1 |
| `bleu_score` | BLEU score for n-gram overlap | 0 to 1 |

### Relevance Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `answer_relevance` | Semantic similarity between question and answer | 0 to 1 |
| `context_relevance` | Relevance of retrieved contexts to question | 0 to 1 |

### Faithfulness Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `faithfulness_score` | How faithful the answer is to contexts | 0 to 1 |
| `hallucination_score` | Degree of unsupported content (lower is better) | 0 to 1 |
| `semantic_grounding` | Semantic overlap with contexts | 0 to 1 |

## Creating Evaluation Datasets

### JSON Format

```json
[
  {
    "id": "eval_001",
    "question": "What are the most common attack types?",
    "ground_truth_answer": "The most common attack types include...",
    "question_type": "log_analysis",
    "difficulty": "medium",
    "expected_contexts": ["context1", "context2"],
    "metadata": {
      "topic": "attack_detection",
      "tags": ["security", "logs"]
    }
  }
]
```

### Required Fields

- `id`: Unique identifier for the sample
- `question`: The question to evaluate
- `ground_truth_answer`: The expected correct answer

### Optional Fields

- `question_type`: Type of question (`log_analysis`, `cybersecurity_knowledge`)
- `difficulty`: Difficulty level (`easy`, `medium`, `hard`)
- `expected_contexts`: List of expected relevant contexts (for precision/recall)
- `metadata`: Additional metadata for analysis

### CSV Format

You can also use CSV format with the following columns:
- id
- question
- ground_truth_answer
- question_type
- difficulty
- expected_contexts (JSON string)
- metadata (JSON string)

## Configuration Options

The evaluation system can be configured via YAML file:

```yaml
# config/evaluation_config.yaml

evaluation:
  batch_size: 1
  max_iterations: 3
  enable_instrumentation: true

metrics:
  speed: true
  accuracy: true
  relevance: true
  faithfulness: true

output:
  directory: "./evaluation_reports"
  formats:
    - json
    - csv
    - html
    - markdown

thresholds:
  min_semantic_similarity: 0.5
  min_faithfulness: 0.5
  max_execution_time: 60.0
```

## Report Formats

### JSON Report
Machine-readable format with complete details:
- All individual results
- Aggregate statistics
- Full execution data

### CSV Report
Spreadsheet-friendly format for analysis:
- One row per sample
- Key metrics as columns
- Easy to import into Excel/Google Sheets

### HTML Report
Human-readable format with visualizations:
- Summary cards
- Metric charts
- Per-sample breakdown
- Insights and recommendations

### Markdown Report
Summary format for documentation:
- Aggregate statistics
- Per-type/difficulty breakdowns
- Insights

## Best Practices

### Dataset Creation

1. **Diversity**: Include questions of varying types and difficulties
2. **Ground Truth Quality**: Ensure ground truth answers are accurate and complete
3. **Edge Cases**: Include edge cases like irrelevant questions
4. **Balance**: Balance between different question types

### Continuous Evaluation

1. **Regular Runs**: Run evaluations regularly (e.g., after model updates)
2. **Track Trends**: Monitor metrics over time
3. **Threshold Alerts**: Set up alerts when metrics fall below thresholds
4. **A/B Testing**: Compare different configurations

### Interpreting Results

1. **Semantic Similarity > 0.7**: Good answer quality
2. **Semantic Similarity 0.5-0.7**: Acceptable but could improve
3. **Semantic Similarity < 0.5**: Poor answer quality, needs investigation

4. **Faithfulness > 0.8**: Answers are well-grounded
5. **Faithfulness < 0.5**: Potential hallucination issues

6. **Execution Time**: 
   - < 10s: Excellent
   - 10-30s: Good
   - > 30s: May need optimization

## Troubleshooting

### Common Issues

**ImportError: Module not found**
```bash
# Ensure dependencies are installed
uv sync
```

**Dataset file not found**
```bash
# Check the path is correct
ls -la data/evaluation/
```

**Timeout errors**
- Increase `query_timeout` in config
- Check network connectivity to Neo4j/MCP servers

**Low accuracy scores**
- Review ground truth answers
- Check retrieval quality
- Examine synthesis prompts

## API Reference

### EvaluationDataset

```python
# Load from JSON
dataset = EvaluationDataset.from_json("path/to/dataset.json")

# Load from CSV
dataset = EvaluationDataset.from_csv("path/to/dataset.csv")

# Filter by type
log_samples = dataset.get_samples_by_type("log_analysis")

# Filter by difficulty
hard_samples = dataset.get_samples_by_difficulty("hard")

# Get summary
summary = dataset.summary()
```

### Evaluator

```python
# Create evaluator
evaluator = Evaluator(config={"batch_size": 2})

# Evaluate single sample
result = await evaluator.evaluate_sample(sample)

# Evaluate dataset
results = await evaluator.evaluate_dataset(dataset)

# Generate summary
summary = evaluator.generate_summary(dataset)
```

### EvaluationReporter

```python
# Create reporter
reporter = EvaluationReporter(results, summary, output_dir="./reports")

# Generate individual reports
json_path = reporter.generate_json_report()
csv_path = reporter.generate_csv_report()
html_path = reporter.generate_html_report()
md_path = reporter.generate_markdown_report()

# Generate all reports
all_paths = reporter.generate_all_reports()
```

### Metrics Functions

```python
from src.evaluation.metrics import (
    calculate_speed_metrics,
    calculate_accuracy_metrics,
    calculate_relevance_metrics,
    calculate_faithfulness,
)

# Calculate individual metrics
speed = calculate_speed_metrics(execution_data)
accuracy = calculate_accuracy_metrics(predicted, ground_truth)
relevance = calculate_relevance_metrics(question, answer, contexts)
faithfulness = calculate_faithfulness(answer, contexts)
```

## Extending the System

### Adding New Metrics

1. Add metric function to `src/evaluation/metrics.py`
2. Call it from `Evaluator.evaluate_sample()`
3. Include in report generation

### Custom Report Formats

1. Add method to `EvaluationReporter`
2. Implement format-specific serialization
3. Add to `generate_all_reports()` if needed
