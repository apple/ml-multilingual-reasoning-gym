"""
Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""

"""Iteratively synthesizes a string by inserting characters according to a pattern.

https://github.com/yongchao98/CodeSteer-v1.0/blob/main/create_dataset/create_dataset_string_synthesis.py
"""

from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import register_dataset
from ..multilingual.base_classes import MultilingualProceduralDataset
from ..config import DatasetConfig

DATASET_NAME = "string_synthesis"

@dataclass
class StringSynthesisConfig(DatasetConfig):
    """Configuration for String Synthesis dataset generation"""

    min_initial_blocks: int = 0  # Minimum number of initial blocks
    max_initial_blocks: int = 5  # Maximum number of initial blocks
    max_iterations: int = 1_000  # Maximum number of iterations to apply the rules (Safety check for infinite loops)
    languages: list[str] | str = "en"
    language_weights: Optional[list[float]] = None

    def validate(self):
        """Validate configuration parameters"""
        assert 0 <= self.min_initial_blocks, "min_initial_blocks must be non-negative"
        assert (
            self.min_initial_blocks <= self.max_initial_blocks
        ), "min_initial_blocks must be less than or equal to max_initial_blocks"
        assert 0 < self.max_iterations, "max_iterations must be positive"

class StringSynthesisDataset(MultilingualProceduralDataset):
    """Generates String Synthesis exercises with configurable difficulty"""

    def __init__(self, config: StringSynthesisConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _apply_rule(self, counts: list[int]) -> list[int]:
        """
        Apply the first applicable rule to the given counts.
        In case no rule is applicable, the counts are returned unchanged.
        """
        # label the indices for the counts
        A_square, B_square, C_square, A_curly, B_curly, C_curly, A_round, B_round, C_round = range(9)
        # Rule 1: One [A], one [B], and one [C] can be combined to form one {A}
        if counts[A_square] >= 1 and counts[B_square] >= 1 and counts[C_square] >= 1:
            counts[A_square] -= 1
            counts[B_square] -= 1
            counts[C_square] -= 1
            counts[A_curly] += 1
        # Rule 2: One [A] and one [B] can be combined to form one {C}
        elif counts[A_square] >= 1 and counts[B_square] >= 1:
            counts[A_square] -= 1
            counts[B_square] -= 1
            counts[C_curly] += 1
        # Rule 3: One [B] and one [C] can be combined to form one {B}
        elif counts[B_square] >= 1 and counts[C_square] >= 1:
            counts[B_square] -= 1
            counts[C_square] -= 1
            counts[B_curly] += 1
        # Rule 4: Two [C] can be combined to form one {C}
        elif counts[C_square] >= 2:
            counts[C_square] -= 2
            counts[C_curly] += 1
        # Rule 5: One {A} and one {C} can be combined to form one (A) and one (B)
        elif counts[A_curly] >= 1 and counts[C_curly] >= 1:
            counts[A_curly] -= 1
            counts[C_curly] -= 1
            counts[A_round] += 1
            counts[B_round] += 1
        # Rule 6: Two {B} can be combined to form one (C)
        elif counts[B_curly] >= 2:
            counts[B_curly] -= 2
            counts[C_round] += 1
        return counts

    def _get_answer(self, A_square: int, B_square: int, C_square: int) -> list[list[int]]:
        """Calculate the answer for a given input"""
        # [A] [B] [C] {A} {B} {C} (A) (B) (C)
        counts = [A_square, B_square, C_square] + [0 for _ in range(6)]
        states = [counts]

        for _ in range(self.config.max_iterations):
            new_counts = self._apply_rule(counts[:])
            if new_counts in states:
                break
            states.append(new_counts)
            counts = new_counts

        return states

    def _generate_item(self, idx: int, language: str) -> dict[str, Any]:
        """Generate a single String Synthesis question in the specified language"""
        rng = Random(self.seed + idx)

        A_square = rng.randint(self.config.min_initial_blocks, self.config.max_initial_blocks)
        B_square = rng.randint(self.config.min_initial_blocks, self.config.max_initial_blocks)
        C_square = rng.randint(self.config.min_initial_blocks, self.config.max_initial_blocks)

        states = self._get_answer(A_square, B_square, C_square)
        answer = states[-1]
        answer_str = " ".join(str(x) for x in answer)

        question = self._get_translation(
            "question_template",
            language,
            A_square=A_square,
            B_square=B_square,
            C_square=C_square,
            max_iterations=self.config.max_iterations,
        )

        return {
            "question": question,
            "answer": answer_str,
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "states": states,
                "solution": answer,
                "initial_blocks": (A_square, B_square, C_square),
                "difficulty": {
                    "initial_blocks": (self.config.min_initial_blocks, self.config.max_initial_blocks),
                },
                "language": language,
            },
        }

class StringSynthesisCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(StringSynthesisCurriculum.__name__, StringSynthesisConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="initial_blocks",
                levels=[10, 50, 100, 500],
                description="Number of initial blocks",
                lower_field_name="min_initial_blocks",
                upper_field_name="max_initial_blocks",
                ensure_interval=True,
            )
        )

register_dataset(DATASET_NAME, StringSynthesisDataset, StringSynthesisConfig, StringSynthesisCurriculum)
