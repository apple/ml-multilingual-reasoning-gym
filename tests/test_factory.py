"""
Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""

"""Tests for factory functionality including name:label support"""

import pytest
from dataclasses import dataclass

from reasoning_gym.config import DatasetConfig
from reasoning_gym.dataset import ProceduralDataset
from reasoning_gym.factory import register_dataset, create_dataset, DATASETS


@dataclass
class MockDatasetConfig(DatasetConfig):
    """Mock configuration for factory tests"""
    difficulty: str = "easy"


class MockDataset(ProceduralDataset):
    """Mock dataset for factory tests"""
    
    def __init__(self, config: MockDatasetConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)
    
    def __getitem__(self, idx: int) -> dict:
        return {
            "question": f"Test question {idx} with difficulty {self.config.difficulty}",
            "answer": f"answer_{idx}_{self.config.difficulty}",
            "metadata": {
                "source_dataset": "test_dataset",
                "source_index": idx,
                "difficulty": self.config.difficulty
            }
        }


def test_register_dataset_simple_name():
    """Test registering dataset with simple name"""
    register_dataset("test_simple", MockDataset, MockDatasetConfig)
    
    assert "test_simple" in DATASETS
    dataset_cls, config_cls = DATASETS["test_simple"]
    assert dataset_cls == MockDataset
    assert config_cls == MockDatasetConfig


def test_register_dataset_labeled_name():
    """Test registering dataset with name:label format"""
    register_dataset("test_labeled:easy", MockDataset, MockDatasetConfig)
    register_dataset("test_labeled:hard", MockDataset, MockDatasetConfig)
    
    assert "test_labeled:easy" in DATASETS
    assert "test_labeled:hard" in DATASETS
    
    # Both should have same classes but are separate registrations
    easy_cls, easy_config = DATASETS["test_labeled:easy"]
    hard_cls, hard_config = DATASETS["test_labeled:hard"]
    
    assert easy_cls == MockDataset == hard_cls
    assert easy_config == MockDatasetConfig == hard_config


def test_create_dataset_with_labels():
    """Test creating datasets with different labels"""
    register_dataset("test_multi:beginner", MockDataset, MockDatasetConfig)
    register_dataset("test_multi:expert", MockDataset, MockDatasetConfig)
    
    # Create datasets with different configurations
    beginner_dataset = create_dataset("test_multi:beginner", difficulty="beginner", size=5, seed=42)
    expert_dataset = create_dataset("test_multi:expert", difficulty="expert", size=3, seed=123)
    
    assert beginner_dataset.config.difficulty == "beginner"
    assert beginner_dataset.config.size == 5
    assert beginner_dataset.config.seed == 42
    
    assert expert_dataset.config.difficulty == "expert"
    assert expert_dataset.config.size == 3
    assert expert_dataset.config.seed == 123
    
    # Verify they generate different content
    beginner_item = beginner_dataset[0]
    expert_item = expert_dataset[0]
    
    assert "beginner" in beginner_item["question"]
    assert "expert" in expert_item["question"]
    assert beginner_item["answer"] != expert_item["answer"]


def test_register_duplicate_name_raises_error():
    """Test that registering duplicate names raises ValueError"""
    register_dataset("test_duplicate", MockDataset, MockDatasetConfig)
    
    with pytest.raises(ValueError, match="Dataset 'test_duplicate' is already registered"):
        register_dataset("test_duplicate", MockDataset, MockDatasetConfig)


def test_register_duplicate_labeled_name_raises_error():
    """Test that registering duplicate labeled names raises ValueError"""
    register_dataset("test_dup_label:version1", MockDataset, MockDatasetConfig)
    
    with pytest.raises(ValueError, match="Dataset 'test_dup_label:version1' is already registered"):
        register_dataset("test_dup_label:version1", MockDataset, MockDatasetConfig)


def test_same_base_name_different_labels_allowed():
    """Test that same base name with different labels is allowed"""
    # This should not raise any errors
    register_dataset("test_base:v1", MockDataset, MockDatasetConfig)
    register_dataset("test_base:v2", MockDataset, MockDatasetConfig) 
    register_dataset("test_base:v3", MockDataset, MockDatasetConfig)
    
    # All should be registered
    assert "test_base:v1" in DATASETS
    assert "test_base:v2" in DATASETS
    assert "test_base:v3" in DATASETS


def test_create_nonexistent_dataset_raises_error():
    """Test that creating non-existent dataset raises ValueError"""
    with pytest.raises(ValueError, match="Dataset 'nonexistent' not registered"):
        create_dataset("nonexistent")
    
    with pytest.raises(ValueError, match="Dataset 'nonexistent:label' not registered"):
        create_dataset("nonexistent:label")


def test_labeled_datasets_are_independent():
    """Test that labeled datasets with same base name are independent"""
    @dataclass
    class ConfigurableMockConfig(DatasetConfig):
        multiplier: int = 1
    
    class ConfigurableMockDataset(ProceduralDataset):
        def __init__(self, config: ConfigurableMockConfig):
            super().__init__(config=config, seed=config.seed, size=config.size)
        
        def __getitem__(self, idx: int) -> dict:
            return {
                "question": f"Question {idx}",
                "answer": str(idx * self.config.multiplier),
                "metadata": {"source_dataset": "configurable_test", "source_index": idx}
            }
    
    register_dataset("configurable:x2", ConfigurableMockDataset, ConfigurableMockConfig)
    register_dataset("configurable:x5", ConfigurableMockDataset, ConfigurableMockConfig)
    
    dataset_x2 = create_dataset("configurable:x2", multiplier=2, size=3, seed=42)
    dataset_x5 = create_dataset("configurable:x5", multiplier=5, size=3, seed=42)
    
    # Same index, different results due to different multipliers
    item_x2 = dataset_x2[2]
    item_x5 = dataset_x5[2]
    
    assert item_x2["answer"] == "4"  # 2 * 2
    assert item_x5["answer"] == "10"  # 2 * 5
    assert item_x2["question"] == item_x5["question"]  # Same question template


# Clean up registered datasets after tests
def teardown_module():
    """Clean up test datasets from registry"""
    test_keys = [key for key in DATASETS.keys() if key.startswith("test_") or key.startswith("configurable")]
    for key in test_keys:
        del DATASETS[key]