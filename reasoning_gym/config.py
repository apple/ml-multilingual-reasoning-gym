"""
Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""

"""Dataset configuration base classes for reasoning gym."""

from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class DatasetConfig:
    """Base configuration class for all dataset configs.

    This class defines the required parameters that all dataset configs must have:
    - size: The number of examples to generate
    - seed: Random seed for reproducible generation
    - languages: Language(s) for the dataset (single language or list)
    - language_weights: Distribution weights for multiple languages (uniform if None)
    """

    size: int = 500
    seed: Optional[int] = None
    languages: Union[str, List[str]] = "en"
    language_weights: Optional[List[float]] = None

    def validate(self) -> None:
        """Validate configuration parameters."""
        pass