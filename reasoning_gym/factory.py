"""
Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""

from dataclasses import is_dataclass
from typing import Callable, Optional, Type, TypeVar

from reasoning_gym.coaching.base_curriculum import BaseCurriculum, ConfigT

from .dataset import ProceduralDataset
from .utils import maybe_decompose_dataset_name

# Type variables for generic type hints

DatasetT = TypeVar("DatasetT", bound=ProceduralDataset)
CurriculumT = TypeVar("CurriculumT", bound=BaseCurriculum)

# Global registry of datasets
DATASETS: dict[str, tuple[Type[ProceduralDataset], Type]] = {}
CURRICULA: dict[str, BaseCurriculum] = {}


def register_dataset(
    name: str,
    dataset_cls: Type[DatasetT],
    config_cls: Type[ConfigT],
    curriculum_cls: Optional[CurriculumT] = None,
) -> None:
    """
    Register a dataset class with its configuration class and optional curriculum.

    Supports both simple names ("maze") and labeled names ("maze:easy", "maze:hard")
    for registering multiple variants of the same dataset with different configurations.

    Args:
        name: Unique identifier for the dataset. Format: "name" or "name:label"
        dataset_cls: Class derived from ProceduralDataset
        config_cls: Configuration dataclass for the dataset
        curriculum_cls: Optional curriculum class for progressive difficulty

    Raises:
        ValueError: If name is already registered or invalid types provided
    """
    if name in DATASETS:
        raise ValueError(f"Dataset '{name}' is already registered")

    if not issubclass(dataset_cls, ProceduralDataset):
        raise ValueError(f"Dataset class must inherit from ProceduralDataset, got {dataset_cls}")

    if not is_dataclass(config_cls):
        raise ValueError(f"Config class must be a dataclass, got {config_cls}")

    DATASETS[name] = (dataset_cls, config_cls)

    if curriculum_cls:
        CURRICULA[name] = curriculum_cls


def create_dataset(name: str, **kwargs) -> ProceduralDataset:
    """
    Create a dataset instance by name with the given configuration.

    Args:
        name: Registered dataset name

    Returns:
        Configured dataset instance

    Raises:
        ValueError: If dataset not found or config type mismatch
    """
    # Decompose dataset name to handle language suffixes
    original_name = name
    name, languages = maybe_decompose_dataset_name(name, DATASETS)
    if name not in DATASETS:
        raise ValueError(f"Dataset '{original_name}' not registered")

    dataset_cls, config_cls = DATASETS[name]

    # Add languages to config if suffix was provided
    if languages:
        if kwargs.get("languages"):
            print(f"Overwriting languages for dataset {name} from {kwargs['languages']} to {languages}")
        kwargs["languages"] = languages
    config = config_cls(**kwargs)
    if hasattr(config, "validate"):
        config.validate()

    return dataset_cls(config=config)


def create_curriculum(name: str) -> BaseCurriculum:
    """
    Create a curriculum instance for the named dataset.

    Args:
        name: Registered dataset name

    Returns:
        Configured curriculum instance

    Raises:
        ValueError: If dataset not found or has no curriculum registered
    """
    # Decompose dataset name to handle language suffixes
    name, _ = maybe_decompose_dataset_name(name, CURRICULA)

    if name not in CURRICULA:
        raise ValueError(f"No curriculum registered for dataset '{name}'")

    curriculum_cls = CURRICULA[name]

    return curriculum_cls()


def has_curriculum(name: str) -> bool:
    """Check if a curriculum is registered for the dataset (base name)"""
    name, _ = maybe_decompose_dataset_name(name, CURRICULA)
    return name in CURRICULA


def get_score_answer_fn(name: str) -> Callable[[], float]:
    """
    Get the score answer function for the named dataset.

    Args:
        name: Registered dataset name

    Returns:
        Score function for the dataset

    Raises:
        ValueError: If dataset not found
    """
    # Decompose dataset name to handle language suffixes
    name, languages = maybe_decompose_dataset_name(name, DATASETS)
    if name not in DATASETS:
        raise ValueError(f"Dataset '{name}' not registered")

    dataset_cls, config_cls = DATASETS[name]

    config_kwargs = {}
    if languages:
        config_kwargs["languages"] = languages

    return dataset_cls(config=config_cls(**config_kwargs)).score_answer
