# src/evaluation/reporter.py
"""
Report generation for evaluation results.
Supports JSON, CSV, HTML, and Markdown formats.
"""

import json
import csv
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

from src.evaluation.evaluator import EvaluationResult, EvaluationSummary


class EvaluationReporter:
    """
    Generates detailed evaluation reports in multiple formats.
    """
    
    def __init__(
        self,
        results: list[EvaluationResult],
        summary: EvaluationSummary,
        output_dir: Optional[str | Path] = None,
    ):
        """
        Initialize the reporter.
        
        Args:
            results: List of individual evaluation results
            summary: Aggregated evaluation summary
            output_dir: Directory to save reports
        """
        self.results = results
        self.summary = summary
        self.output_dir = Path(output_dir) if output_dir else Path("./evaluation_reports")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _ensure_output_dir(self) -> None:
        """Create output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_json_report(self, filename: Optional[str] = None) -> Path:
        """
        Generate detailed JSON report.
        
        Args:
            filename: Optional custom filename
            
        Returns:
            Path to generated file
        """
        self._ensure_output_dir()
        
        filename = filename or f"evaluation_report_{self.timestamp}.json"
        filepath = self.output_dir / filename
        
        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_samples": self.summary.total_samples,
                "successful_evaluations": self.summary.successful_evaluations,
                "failed_evaluations": self.summary.failed_evaluations,
            },
            "summary": self.summary.to_dict(),
            "results": [r.to_dict() for r in self.results],
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        return filepath
    
    def generate_csv_report(self, filename: Optional[str] = None) -> Path:
        """
        Generate CSV report for spreadsheet analysis.
        
        Args:
            filename: Optional custom filename
            
        Returns:
            Path to generated file
        """
        self._ensure_output_dir()
        
        filename = filename or f"evaluation_report_{self.timestamp}.csv"
        filepath = self.output_dir / filename
        
        # Flatten results for CSV
        fieldnames = [
            "sample_id",
            "question",
            "ground_truth",
            "predicted_answer",
            "error",
            # Speed metrics
            "total_execution_time_seconds",
            # Accuracy metrics
            "exact_match",
            "semantic_similarity",
            "rouge1",
            "rouge2",
            "rougeL",
            "bleu_score",
            # Relevance metrics
            "answer_relevance",
            "context_relevance_mean",
            # Faithfulness metrics
            "faithfulness_score",
            "hallucination_score",
        ]
        
        with open(filepath, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                row = {
                    "sample_id": result.sample_id,
                    "question": result.question[:200],  # Truncate for readability
                    "ground_truth": result.ground_truth[:200],
                    "predicted_answer": result.predicted_answer[:200],
                    "error": result.error or "",
                    "total_execution_time_seconds": result.speed_metrics.get("total_execution_time_seconds", ""),
                    "exact_match": result.accuracy_metrics.get("exact_match", ""),
                    "semantic_similarity": result.accuracy_metrics.get("semantic_similarity", ""),
                    "rouge1": result.accuracy_metrics.get("rouge_scores", {}).get("rouge1", ""),
                    "rouge2": result.accuracy_metrics.get("rouge_scores", {}).get("rouge2", ""),
                    "rougeL": result.accuracy_metrics.get("rouge_scores", {}).get("rougeL", ""),
                    "bleu_score": result.accuracy_metrics.get("bleu_score", ""),
                    "answer_relevance": result.relevance_metrics.get("answer_relevance", ""),
                    "context_relevance_mean": result.relevance_metrics.get("context_relevance", {}).get("mean", ""),
                    "faithfulness_score": result.faithfulness_metrics.get("faithfulness_score", ""),
                    "hallucination_score": result.faithfulness_metrics.get("hallucination_detection", {}).get("hallucination_score", ""),
                }
                writer.writerow(row)
        
        return filepath
    
    def generate_html_report(self, filename: Optional[str] = None) -> Path:
        """
        Generate human-readable HTML report with visualizations.
        
        Args:
            filename: Optional custom filename
            
        Returns:
            Path to generated file
        """
        self._ensure_output_dir()
        
        filename = filename or f"evaluation_report_{self.timestamp}.html"
        filepath = self.output_dir / filename
        
        html_content = self._build_html_report()
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        return filepath
    
    def _build_html_report(self) -> str:
        """Build the HTML report content."""
        # Calculate some statistics for visualizations
        valid_results = [r for r in self.results if r.error is None]
        
        # Extract metrics for charts
        semantic_sims = [r.accuracy_metrics.get("semantic_similarity", 0) for r in valid_results]
        exec_times = [r.speed_metrics.get("total_execution_time_seconds", 0) for r in valid_results]
        faithfulness = [r.faithfulness_metrics.get("faithfulness_score", 0) for r in valid_results]
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluation Report - {self.timestamp}</title>
    <style>
        :root {{
            --primary-color: #2563eb;
            --success-color: #16a34a;
            --warning-color: #d97706;
            --danger-color: #dc2626;
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --border-color: #e2e8f0;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        h1 {{
            color: var(--primary-color);
            margin-bottom: 0.5rem;
        }}
        
        .subtitle {{
            color: var(--text-secondary);
            margin-bottom: 2rem;
        }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        
        .card {{
            background: var(--card-bg);
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid var(--border-color);
        }}
        
        .card-title {{
            font-size: 0.875rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }}
        
        .card-value {{
            font-size: 2rem;
            font-weight: 600;
            color: var(--text-primary);
        }}
        
        .card-value.success {{
            color: var(--success-color);
        }}
        
        .card-value.warning {{
            color: var(--warning-color);
        }}
        
        .card-value.danger {{
            color: var(--danger-color);
        }}
        
        .section {{
            background: var(--card-bg);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid var(--border-color);
        }}
        
        .section-title {{
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: var(--text-primary);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}
        
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        
        th {{
            background-color: var(--bg-color);
            font-weight: 600;
            color: var(--text-secondary);
        }}
        
        tr:hover {{
            background-color: var(--bg-color);
        }}
        
        .metric-bar {{
            height: 8px;
            background: var(--border-color);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 0.25rem;
        }}
        
        .metric-bar-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }}
        
        .metric-bar-fill.good {{
            background: var(--success-color);
        }}
        
        .metric-bar-fill.medium {{
            background: var(--warning-color);
        }}
        
        .metric-bar-fill.poor {{
            background: var(--danger-color);
        }}
        
        .truncate {{
            max-width: 300px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 500;
        }}
        
        .badge.success {{
            background: #dcfce7;
            color: var(--success-color);
        }}
        
        .badge.error {{
            background: #fee2e2;
            color: var(--danger-color);
        }}
        
        .insights {{
            background: #eff6ff;
            border-left: 4px solid var(--primary-color);
            padding: 1rem;
            margin-top: 1rem;
        }}
        
        .insights-title {{
            font-weight: 600;
            margin-bottom: 0.5rem;
        }}
        
        .insights ul {{
            margin-left: 1.5rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Multi-Agent RAG Evaluation Report</h1>
        <p class="subtitle">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <!-- Summary Cards -->
        <div class="grid">
            <div class="card">
                <div class="card-title">Total Samples</div>
                <div class="card-value">{self.summary.total_samples}</div>
            </div>
            <div class="card">
                <div class="card-title">Successful</div>
                <div class="card-value success">{self.summary.successful_evaluations}</div>
            </div>
            <div class="card">
                <div class="card-title">Failed</div>
                <div class="card-value {'danger' if self.summary.failed_evaluations > 0 else ''}">{self.summary.failed_evaluations}</div>
            </div>
            <div class="card">
                <div class="card-title">Success Rate</div>
                <div class="card-value">{self.summary.successful_evaluations / max(self.summary.total_samples, 1) * 100:.1f}%</div>
            </div>
        </div>
        
        <!-- Aggregate Metrics -->
        <div class="section">
            <h2 class="section-title">Aggregate Metrics</h2>
            <div class="grid">
                <div class="card">
                    <div class="card-title">Avg Execution Time</div>
                    <div class="card-value">{self.summary.aggregate_speed.get('mean_total_time', 0):.2f}s</div>
                </div>
                <div class="card">
                    <div class="card-title">Avg Semantic Similarity</div>
                    <div class="card-value">{self.summary.aggregate_accuracy.get('mean_semantic_similarity', 0):.3f}</div>
                    <div class="metric-bar">
                        <div class="metric-bar-fill {'good' if self.summary.aggregate_accuracy.get('mean_semantic_similarity', 0) > 0.7 else 'medium' if self.summary.aggregate_accuracy.get('mean_semantic_similarity', 0) > 0.5 else 'poor'}" 
                             style="width: {self.summary.aggregate_accuracy.get('mean_semantic_similarity', 0) * 100:.0f}%"></div>
                    </div>
                </div>
                <div class="card">
                    <div class="card-title">Avg Answer Relevance</div>
                    <div class="card-value">{self.summary.aggregate_relevance.get('mean_answer_relevance', 0):.3f}</div>
                    <div class="metric-bar">
                        <div class="metric-bar-fill {'good' if self.summary.aggregate_relevance.get('mean_answer_relevance', 0) > 0.7 else 'medium' if self.summary.aggregate_relevance.get('mean_answer_relevance', 0) > 0.5 else 'poor'}" 
                             style="width: {self.summary.aggregate_relevance.get('mean_answer_relevance', 0) * 100:.0f}%"></div>
                    </div>
                </div>
                <div class="card">
                    <div class="card-title">Avg Faithfulness</div>
                    <div class="card-value">{self.summary.aggregate_faithfulness.get('mean_faithfulness', 0):.3f}</div>
                    <div class="metric-bar">
                        <div class="metric-bar-fill {'good' if self.summary.aggregate_faithfulness.get('mean_faithfulness', 0) > 0.7 else 'medium' if self.summary.aggregate_faithfulness.get('mean_faithfulness', 0) > 0.5 else 'poor'}" 
                             style="width: {self.summary.aggregate_faithfulness.get('mean_faithfulness', 0) * 100:.0f}%"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Per-Sample Results -->
        <div class="section">
            <h2 class="section-title">Individual Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Question</th>
                        <th>Status</th>
                        <th>Time (s)</th>
                        <th>Semantic Sim</th>
                        <th>Faithfulness</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(self._build_result_row(r) for r in self.results)}
                </tbody>
            </table>
        </div>
        
        <!-- Insights -->
        <div class="section">
            <h2 class="section-title">Insights & Recommendations</h2>
            {self._generate_insights_html()}
        </div>
    </div>
</body>
</html>"""
        
        return html
    
    def _build_result_row(self, result: EvaluationResult) -> str:
        """Build a table row for a single result."""
        status = "error" if result.error else "success"
        status_text = "Failed" if result.error else "Success"
        
        sem_sim = result.accuracy_metrics.get("semantic_similarity", 0)
        faithful = result.faithfulness_metrics.get("faithfulness_score", 0)
        exec_time = result.speed_metrics.get("total_execution_time_seconds", 0)
        
        return f"""
        <tr>
            <td>{result.sample_id}</td>
            <td class="truncate" title="{result.question}">{result.question[:50]}...</td>
            <td><span class="badge {status}">{status_text}</span></td>
            <td>{exec_time:.2f}</td>
            <td>{sem_sim:.3f}</td>
            <td>{faithful:.3f}</td>
        </tr>
        """
    
    def _generate_insights_html(self) -> str:
        """Generate insights based on evaluation results."""
        insights = []
        
        # Check execution time
        avg_time = self.summary.aggregate_speed.get("mean_total_time", 0)
        if avg_time > 30:
            insights.append("High average execution time detected. Consider optimizing the workflow or caching frequent queries.")
        elif avg_time < 5:
            insights.append("Excellent execution speed. The system responds quickly to queries.")
        
        # Check accuracy
        avg_sim = self.summary.aggregate_accuracy.get("mean_semantic_similarity", 0)
        if avg_sim < 0.5:
            insights.append("Low semantic similarity to ground truth. Review retrieval quality and synthesis prompts.")
        elif avg_sim > 0.8:
            insights.append("Strong semantic similarity to ground truth answers.")
        
        # Check faithfulness
        avg_faithful = self.summary.aggregate_faithfulness.get("mean_faithfulness", 0)
        if avg_faithful < 0.5:
            insights.append("Potential hallucination issues detected. Answers may contain unsupported information.")
        elif avg_faithful > 0.8:
            insights.append("High faithfulness score. Answers are well-grounded in retrieved contexts.")
        
        # Check failure rate
        failure_rate = self.summary.failed_evaluations / max(self.summary.total_samples, 1)
        if failure_rate > 0.1:
            insights.append(f"High failure rate ({failure_rate*100:.1f}%). Investigate error patterns.")
        
        if not insights:
            insights.append("System performance is within acceptable ranges across all metrics.")
        
        return f"""
        <div class="insights">
            <div class="insights-title">Key Findings:</div>
            <ul>
                {''.join(f'<li>{insight}</li>' for insight in insights)}
            </ul>
        </div>
        """
    
    def generate_markdown_report(self, filename: Optional[str] = None) -> Path:
        """
        Generate Markdown summary report.
        
        Args:
            filename: Optional custom filename
            
        Returns:
            Path to generated file
        """
        self._ensure_output_dir()
        
        filename = filename or f"evaluation_report_{self.timestamp}.md"
        filepath = self.output_dir / filename
        
        md_content = self._build_markdown_report()
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md_content)
        
        return filepath
    
    def _build_markdown_report(self) -> str:
        """Build the Markdown report content."""
        md = f"""# Multi-Agent RAG Evaluation Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary

| Metric | Value |
|--------|-------|
| Total Samples | {self.summary.total_samples} |
| Successful Evaluations | {self.summary.successful_evaluations} |
| Failed Evaluations | {self.summary.failed_evaluations} |
| Success Rate | {self.summary.successful_evaluations / max(self.summary.total_samples, 1) * 100:.1f}% |

## Speed Metrics

| Metric | Value |
|--------|-------|
| Mean Execution Time | {self.summary.aggregate_speed.get('mean_total_time', 0):.2f}s |
| Median Execution Time | {self.summary.aggregate_speed.get('median_total_time', 0):.2f}s |
| Std Dev | {self.summary.aggregate_speed.get('std_total_time', 0):.2f}s |
| Min Time | {self.summary.aggregate_speed.get('min_total_time', 0):.2f}s |
| Max Time | {self.summary.aggregate_speed.get('max_total_time', 0):.2f}s |

## Accuracy Metrics

| Metric | Value |
|--------|-------|
| Mean Semantic Similarity | {self.summary.aggregate_accuracy.get('mean_semantic_similarity', 0):.4f} |
| Mean ROUGE-1 | {self.summary.aggregate_accuracy.get('mean_rouge1', 0):.4f} |
| Mean BLEU | {self.summary.aggregate_accuracy.get('mean_bleu', 0):.4f} |

## Relevance Metrics

| Metric | Value |
|--------|-------|
| Mean Answer Relevance | {self.summary.aggregate_relevance.get('mean_answer_relevance', 0):.4f} |
| Mean Context Relevance | {self.summary.aggregate_relevance.get('mean_context_relevance', 0):.4f} |

## Faithfulness Metrics

| Metric | Value |
|--------|-------|
| Mean Faithfulness | {self.summary.aggregate_faithfulness.get('mean_faithfulness', 0):.4f} |
| Min Faithfulness | {self.summary.aggregate_faithfulness.get('min_faithfulness', 0):.4f} |

"""
        
        # Add per-type metrics if available
        if self.summary.per_type_metrics:
            md += "## Metrics by Question Type\n\n"
            for qtype, metrics in self.summary.per_type_metrics.items():
                md += f"### {qtype}\n\n"
                md += f"- Mean Execution Time: {metrics.get('speed', {}).get('mean_total_time', 0):.2f}s\n"
                md += f"- Mean Semantic Similarity: {metrics.get('accuracy', {}).get('mean_semantic_similarity', 0):.4f}\n"
                md += f"- Mean Faithfulness: {metrics.get('faithfulness', {}).get('mean_faithfulness', 0):.4f}\n\n"
        
        # Add per-difficulty metrics if available
        if self.summary.per_difficulty_metrics:
            md += "## Metrics by Difficulty Level\n\n"
            for difficulty, metrics in self.summary.per_difficulty_metrics.items():
                md += f"### {difficulty}\n\n"
                md += f"- Mean Execution Time: {metrics.get('speed', {}).get('mean_total_time', 0):.2f}s\n"
                md += f"- Mean Semantic Similarity: {metrics.get('accuracy', {}).get('mean_semantic_similarity', 0):.4f}\n"
                md += f"- Mean Faithfulness: {metrics.get('faithfulness', {}).get('mean_faithfulness', 0):.4f}\n\n"
        
        # Add individual results table
        md += "## Individual Results\n\n"
        md += "| ID | Question | Status | Time (s) | Semantic Sim | Faithfulness |\n"
        md += "|---|---|---|---|---|---|\n"
        
        for result in self.results:
            status = "❌ Failed" if result.error else "✅ Success"
            question = result.question[:40] + "..." if len(result.question) > 40 else result.question
            sem_sim = result.accuracy_metrics.get("semantic_similarity", 0)
            faithful = result.faithfulness_metrics.get("faithfulness_score", 0)
            exec_time = result.speed_metrics.get("total_execution_time_seconds", 0)
            
            md += f"| {result.sample_id} | {question} | {status} | {exec_time:.2f} | {sem_sim:.3f} | {faithful:.3f} |\n"
        
        # Add insights
        md += "\n## Insights & Recommendations\n\n"
        md += self._generate_insights_markdown()
        
        return md
    
    def _generate_insights_markdown(self) -> str:
        """Generate insights in Markdown format."""
        insights = []
        
        # Check execution time
        avg_time = self.summary.aggregate_speed.get("mean_total_time", 0)
        if avg_time > 30:
            insights.append("⚠️ **High execution time** detected. Consider optimizing the workflow or implementing caching.")
        elif avg_time < 5:
            insights.append("✅ **Excellent execution speed**. The system responds quickly to queries.")
        
        # Check accuracy
        avg_sim = self.summary.aggregate_accuracy.get("mean_semantic_similarity", 0)
        if avg_sim < 0.5:
            insights.append("⚠️ **Low accuracy** detected. Review retrieval quality and synthesis prompts.")
        elif avg_sim > 0.8:
            insights.append("✅ **Strong accuracy**. Answers closely match ground truth.")
        
        # Check faithfulness
        avg_faithful = self.summary.aggregate_faithfulness.get("mean_faithfulness", 0)
        if avg_faithful < 0.5:
            insights.append("⚠️ **Potential hallucination issues**. Answers may contain unsupported information.")
        elif avg_faithful > 0.8:
            insights.append("✅ **High faithfulness**. Answers are well-grounded in retrieved contexts.")
        
        if not insights:
            insights.append("ℹ️ System performance is within acceptable ranges across all metrics.")
        
        return "\n".join(f"- {insight}" for insight in insights)
    
    def generate_all_reports(self) -> dict[str, Path]:
        """
        Generate reports in all supported formats.
        
        Returns:
            Dictionary mapping format names to file paths
        """
        return {
            "json": self.generate_json_report(),
            "csv": self.generate_csv_report(),
            "html": self.generate_html_report(),
            "markdown": self.generate_markdown_report(),
        }
