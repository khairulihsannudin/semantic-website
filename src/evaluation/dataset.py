# src/evaluation/dataset.py
"""
Dataset loading and management for evaluation.
"""

import json
import csv
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class EvaluationSample:
    """
    A single evaluation sample.
    
    Attributes:
        id: Unique identifier for the sample
        question: The question to evaluate
        ground_truth_answer: Expected correct answer
        question_type: Type of question (log_analysis / cybersecurity_knowledge)
        difficulty: Difficulty level (easy / medium / hard)
        expected_contexts: Optional list of expected relevant contexts
        metadata: Additional metadata (topic, tags, etc.)
    """
    id: str
    question: str
    ground_truth_answer: str
    question_type: str = "log_analysis"
    difficulty: str = "medium"
    expected_contexts: Optional[list[str]] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationSample":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            question=data.get("question", ""),
            ground_truth_answer=data.get("ground_truth_answer", ""),
            question_type=data.get("question_type", "log_analysis"),
            difficulty=data.get("difficulty", "medium"),
            expected_contexts=data.get("expected_contexts"),
            metadata=data.get("metadata", {}),
        )


class EvaluationDataset:
    """
    Manages evaluation datasets.
    Supports loading from JSON and CSV files.
    """
    
    def __init__(self, samples: Optional[list[EvaluationSample]] = None):
        """
        Initialize the dataset.
        
        Args:
            samples: Optional list of evaluation samples
        """
        self.samples: list[EvaluationSample] = samples or []
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __iter__(self):
        return iter(self.samples)
    
    def __getitem__(self, idx: int) -> EvaluationSample:
        return self.samples[idx]
    
    def add_sample(self, sample: EvaluationSample) -> None:
        """Add a sample to the dataset."""
        self.samples.append(sample)
    
    def get_samples_by_type(self, question_type: str) -> list[EvaluationSample]:
        """Get samples filtered by question type."""
        return [s for s in self.samples if s.question_type == question_type]
    
    def get_samples_by_difficulty(self, difficulty: str) -> list[EvaluationSample]:
        """Get samples filtered by difficulty level."""
        return [s for s in self.samples if s.difficulty == difficulty]
    
    @classmethod
    def from_json(cls, file_path: str | Path) -> "EvaluationDataset":
        """
        Load dataset from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            EvaluationDataset instance
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle both list and dict with "samples" key
        if isinstance(data, dict):
            data = data.get("samples", [])
        
        samples = [EvaluationSample.from_dict(item) for item in data]
        return cls(samples=samples)
    
    @classmethod
    def from_csv(cls, file_path: str | Path) -> "EvaluationDataset":
        """
        Load dataset from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            EvaluationDataset instance
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        samples = []
        with open(file_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse expected_contexts if present (stored as JSON string in CSV)
                expected_contexts = None
                if row.get("expected_contexts"):
                    try:
                        expected_contexts = json.loads(row["expected_contexts"])
                    except json.JSONDecodeError:
                        expected_contexts = None
                
                # Parse metadata if present
                metadata = {}
                if row.get("metadata"):
                    try:
                        metadata = json.loads(row["metadata"])
                    except json.JSONDecodeError:
                        metadata = {}
                
                sample = EvaluationSample(
                    id=row.get("id", ""),
                    question=row.get("question", ""),
                    ground_truth_answer=row.get("ground_truth_answer", ""),
                    question_type=row.get("question_type", "log_analysis"),
                    difficulty=row.get("difficulty", "medium"),
                    expected_contexts=expected_contexts,
                    metadata=metadata,
                )
                samples.append(sample)
        
        return cls(samples=samples)
    
    def to_json(self, file_path: str | Path) -> None:
        """
        Save dataset to JSON file.
        
        Args:
            file_path: Path to save JSON file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = [sample.to_dict() for sample in self.samples]
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def to_csv(self, file_path: str | Path) -> None:
        """
        Save dataset to CSV file.
        
        Args:
            file_path: Path to save CSV file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.samples:
            return
        
        fieldnames = [
            "id", "question", "ground_truth_answer", 
            "question_type", "difficulty", "expected_contexts", "metadata"
        ]
        
        with open(file_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for sample in self.samples:
                row = {
                    "id": sample.id,
                    "question": sample.question,
                    "ground_truth_answer": sample.ground_truth_answer,
                    "question_type": sample.question_type,
                    "difficulty": sample.difficulty,
                    "expected_contexts": json.dumps(sample.expected_contexts) if sample.expected_contexts else "",
                    "metadata": json.dumps(sample.metadata) if sample.metadata else "",
                }
                writer.writerow(row)
    
    def summary(self) -> dict[str, Any]:
        """
        Get summary statistics of the dataset.
        
        Returns:
            Dictionary with summary statistics
        """
        type_counts: dict[str, int] = {}
        difficulty_counts: dict[str, int] = {}
        
        for sample in self.samples:
            type_counts[sample.question_type] = type_counts.get(sample.question_type, 0) + 1
            difficulty_counts[sample.difficulty] = difficulty_counts.get(sample.difficulty, 0) + 1
        
        return {
            "total_samples": len(self.samples),
            "by_type": type_counts,
            "by_difficulty": difficulty_counts,
        }
