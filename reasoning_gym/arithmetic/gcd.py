"""
Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""

"""Greatest Common Divisor (GCD) task generator"""

from dataclasses import dataclass
from functools import reduce
from math import gcd
from random import Random
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import register_dataset
from ..config import DatasetConfig
from ..multilingual.base_classes import MultilingualProceduralDataset

DATASET_NAME = "gcd"

@dataclass
class GCDConfig(DatasetConfig):
    """Configuration for GCD task generation"""

    min_numbers: int = 2  # Minimum numbers to find GCD of
    max_numbers: int = 2  # Maximum numbers to find GCD of
    min_value: int = 1  # Minimum value for each number
    max_value: int = 1000  # Maximum value for each number
    languages: list[str] | str = "en"  # Languages to generate tasks in
    language_weights: Optional[list[float]] = None  # Optional weights for language sampling

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.min_numbers >= 2, "min_numbers must be at least 2"
        assert self.max_numbers >= self.min_numbers, "max_numbers must be >= min_numbers"
        assert self.min_value >= 1, "min_value must be positive"
        assert self.max_value > self.min_value, "max_value must be > min_value"

class GCDDataset(MultilingualProceduralDataset):
    """Generates Greatest Common Divisor (GCD) tasks"""

    def __init__(self, config: GCDConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _generate_numbers(self, rng: Random) -> tuple[list[int], int]:
        """Generate a list of random positive integers and their GCD.
        Will try up to 3 times to find numbers with GCD > 1."""

        # Try up to 3 times to get GCD > 1
        for _ in range(3):
            num_count = rng.randint(self.config.min_numbers, self.config.max_numbers)
            numbers = [rng.randint(self.config.min_value, self.config.max_value) for _ in range(num_count)]
            result = reduce(gcd, numbers)
            if result > 1:
                break

        # Return the last generated numbers, whether they met the criteria or not
        return numbers, result

    def _generate_item(self, idx: int, language: str) -> dict[str, Any]:
        """Generate a single GCD task in the specified language"""
        rng = Random(self.seed + idx)

        numbers, result = self._generate_numbers(rng)
        num_terms = len(numbers)
        
        # Get localized numbers separator
        numbers_separator = self._get_translation("numbers_separator", language)
        
        # Format numbers string with localized separator
        numbers_str = numbers_separator.join(str(n) for n in numbers)

        question = self._get_translation("question_template", language, numbers_str=numbers_str)

        return {
            "question": question,
            "answer": str(result),
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "numbers": numbers,
                "result": result,
                "num_terms": num_terms,
                "language": language,
                "difficulty": {
                    "num_terms": (self.config.min_numbers, self.config.max_numbers),
                    "value": (self.config.min_value, self.config.max_value),
                },
            },
        }

class GCDCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(GCDCurriculum.__name__, GCDConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="num_terms",
                levels=[2, 3, 4, 5],
                description="number of terms",
                lower_field_name="min_numbers",
                upper_field_name="max_numbers",
            ),
            RangeAttributeDefinition(
                name="value",
                levels=[100, 1000, 10000, 100000],
                description="maximum value",
                lower_field_name="min_value",
                upper_field_name="max_value",
                ensure_interval=True,
            ),
        )

register_dataset(DATASET_NAME, GCDDataset, GCDConfig, GCDCurriculum)
