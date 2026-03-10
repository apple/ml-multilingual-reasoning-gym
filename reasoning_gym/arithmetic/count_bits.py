"""
Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""

"""Count number of 1 bits in a number."""

from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import register_dataset
from ..multilingual.base_classes import MultilingualProceduralDataset
from ..config import DatasetConfig

DATASET_NAME = "count_bits"

@dataclass
class CountBitsConfig(DatasetConfig):
    """Configuration for Count Bits dataset generation"""

    min_n: int = 1  # Minimum number to consider
    max_n: int = 2**31 - 1  # Maximum number to consider
    languages: list[str] | str = "en"
    language_weights: Optional[list[float]] = None

    def validate(self):
        """Validate configuration parameters"""
        assert 1 <= self.min_n <= self.max_n, "min_n must be between 1 and max_n"

class CountBitsDataset(MultilingualProceduralDataset):
    """Generates Count Bits exercises with configurable difficulty"""

    def __init__(self, config: CountBitsConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _generate_item(self, idx: int, language: str) -> dict[str, Any]:
        """Generate a single Count Bits question in the specified language"""
        rng = Random(self.seed + idx)

        number = rng.randint(self.config.min_n, self.config.max_n)
        binary = bin(number)[2:]
        answer = binary.count("1")

        question = self._get_translation("question_template", language, number=number)

        return {
            "question": question,
            "answer": str(answer),
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "number": number,
                "solution": answer,
                "binary": binary,
                "n": number,
                "difficulty": {
                    "n": (self.config.min_n, self.config.max_n),
                },
                "language": language,
            },
        }

class CountBitsCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(CountBitsCurriculum.__name__, CountBitsConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="n",
                levels=[10, 1_000, 1_000_000, 100_000_000, 2**31 - 1],
                description="Number to count bits in",
                lower_field_name="min_n",
                upper_field_name="max_n",
                ensure_interval=True,
            ),
        )

register_dataset(DATASET_NAME, CountBitsDataset, CountBitsConfig, CountBitsCurriculum)
