"""
Dataset Loader

Utilities for loading benchmark datasets.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class BenchmarkDataset:
    """A benchmark dataset for evaluation.
    
    Attributes:
        name: Dataset name
        queries: List of queries
        labels: Membership labels (1=member, 0=non-member)
        metadata: Additional metadata
    """
    name: str
    queries: List[str]
    labels: List[int]
    metadata: Dict[str, Any] = None
    
    def __len__(self) -> int:
        return len(self.queries)
    
    def split(
        self,
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> Tuple["BenchmarkDataset", "BenchmarkDataset"]:
        """Split dataset into train and test sets.
        
        Args:
            train_ratio: Ratio of training data
            seed: Random seed
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        np.random.seed(seed)
        indices = np.random.permutation(len(self))
        split_point = int(len(self) * train_ratio)
        
        train_indices = indices[:split_point]
        test_indices = indices[split_point:]
        
        train_dataset = BenchmarkDataset(
            name=f"{self.name}_train",
            queries=[self.queries[i] for i in train_indices],
            labels=[self.labels[i] for i in train_indices],
        )
        
        test_dataset = BenchmarkDataset(
            name=f"{self.name}_test",
            queries=[self.queries[i] for i in test_indices],
            labels=[self.labels[i] for i in test_indices],
        )
        
        return train_dataset, test_dataset


class DatasetLoader:
    """Loads benchmark datasets.
    
    Example:
        >>> loader = DatasetLoader()
        >>> dataset = loader.load_mock(num_samples=100)
        >>> print(f"Loaded {len(dataset)} samples")
    """
    
    @staticmethod
    def load_mock(
        num_samples: int = 100,
        member_ratio: float = 0.5,
        seed: int = 42,
    ) -> BenchmarkDataset:
        """Load mock dataset for testing.
        
        Args:
            num_samples: Number of samples
            member_ratio: Ratio of member samples
            seed: Random seed
            
        Returns:
            BenchmarkDataset with mock data
        """
        np.random.seed(seed)
        
        num_member = int(num_samples * member_ratio)
        num_nonmember = num_samples - num_member
        
        queries = []
        labels = []
        
        # Generate member queries
        for i in range(num_member):
            queries.append(f"What is the solution to problem {i} from the training set?")
            labels.append(1)
        
        # Generate non-member queries
        for i in range(num_nonmember):
            queries.append(f"What is the answer to novel question {i}?")
            labels.append(0)
        
        # Shuffle
        indices = np.random.permutation(num_samples)
        queries = [queries[i] for i in indices]
        labels = [labels[i] for i in indices]
        
        return BenchmarkDataset(
            name="mock",
            queries=queries,
            labels=labels,
            metadata={"seed": seed, "member_ratio": member_ratio},
        )
    
    @staticmethod
    def load_from_file(
        path: str,
        name: Optional[str] = None,
    ) -> BenchmarkDataset:
        """Load dataset from file.
        
        Args:
            path: Path to dataset file (JSON or CSV)
            name: Dataset name (default: filename)
            
        Returns:
            BenchmarkDataset
        """
        import json
        from pathlib import Path
        
        path = Path(path)
        if name is None:
            name = path.stem
        
        if path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
            
            return BenchmarkDataset(
                name=name,
                queries=data.get("queries", []),
                labels=data.get("labels", []),
                metadata=data.get("metadata"),
            )
        
        elif path.suffix == ".csv":
            import csv
            
            queries = []
            labels = []
            
            with open(path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    queries.append(row.get("query", ""))
                    labels.append(int(row.get("label", 0)))
            
            return BenchmarkDataset(
                name=name,
                queries=queries,
                labels=labels,
            )
        
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    @staticmethod
    def load_prm800k_sample(
        num_samples: int = 100,
        seed: int = 42,
    ) -> BenchmarkDataset:
        """Load a sample from PRM800K dataset.
        
        Note: This generates synthetic data mimicking PRM800K structure.
        For actual PRM800K data, use load_from_file with the real dataset.
        
        Args:
            num_samples: Number of samples
            seed: Random seed
            
        Returns:
            BenchmarkDataset with PRM800K-like data
        """
        np.random.seed(seed)
        
        math_templates = [
            "Solve for x: {}x + {} = {}",
            "What is the integral of {}x^{} dx?",
            "Find the derivative of x^{} + {}x + {}",
            "Calculate the sum: {} + {} + {} + {}",
            "What is {} × {} + {}?",
        ]
        
        queries = []
        labels = []
        
        for i in range(num_samples):
            template = np.random.choice(math_templates)
            nums = np.random.randint(1, 20, size=10)
            query = template.format(*nums)
            queries.append(query)
            labels.append(int(np.random.random() < 0.5))
        
        return BenchmarkDataset(
            name="prm800k_sample",
            queries=queries,
            labels=labels,
            metadata={"seed": seed, "synthetic": True},
        )
