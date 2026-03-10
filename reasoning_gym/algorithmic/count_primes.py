"""
Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""

"""Count prime numbers in a given interval.

Solution obtained with Sieve of Eratosthenes:
https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes
"""

import math
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import register_dataset
from ..multilingual.base_classes import MultilingualProceduralDataset
from ..config import DatasetConfig

DATASET_NAME = "count_primes"

@dataclass
class CountPrimesConfig(DatasetConfig):
    """Configuration for Count Primes dataset generation"""

    min_n: int = 1  # Lower bound for the interval
    max_n: int = 10_000  # Upper bound for the interval
    languages: list[str] | str = "en"
    language_weights: Optional[list[float]] = None

    def validate(self):
        """Validate configuration parameters"""
        assert 1 <= self.min_n, "min_n must be at least 1"
        assert self.min_n <= self.max_n, "min_n must be less than or equal to max_n"

class CountPrimesDataset(MultilingualProceduralDataset):
    """Generates Count Primes exercises with configurable difficulty"""

    def __init__(self, config: CountPrimesConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)
        self.primes = self._get_primes(config.max_n + 1)

    def _get_primes(self, n: int) -> list[bool]:
        if n <= 1:
            return []
        primes = [True] * n
        primes[0] = primes[1] = False
        for i in range(2, int(math.sqrt(n)) + 1):
            if primes[i]:
                for j in range(2 * i, n, i):
                    primes[j] = False
        return primes

    def _generate_item(self, idx: int, language: str) -> dict[str, Any]:
        """Generate a single Count Primes question in the specified language"""
        rng = Random(self.seed + idx)
        start = rng.randint(self.config.min_n, self.config.max_n)
        end = rng.randint(start, self.config.max_n)
        primes = [i for i in range(start, end + 1) if self.primes[i]]
        answer = len(primes)
        
        question = self._get_translation("question_template", language, start=start, end=end)
        
        return {
            "question": question,
            "answer": str(answer),
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "start": start,
                "end": end,
                "primes": primes,
                "solution": answer,
                "n": (start, end),
                "difficulty": {
                    "n": (self.config.min_n, self.config.max_n),
                },
                "language": language,
            },
        }

class CountPrimesCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(CountPrimesCurriculum.__name__, CountPrimesConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="n",
                levels=[10, 1000, 10_000, 50_000, 100_000],
                description="Up to which number to consider the primes",
                lower_field_name="min_n",
                upper_field_name="max_n",
                ensure_interval=True,
            )
        )

register_dataset(DATASET_NAME, CountPrimesDataset, CountPrimesConfig, CountPrimesCurriculum)
