"""
Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""

"""Iteratively synthesize new machines and parts from existing ones using a set of rules.

https://github.com/yongchao98/CodeSteer-v1.0/blob/main/create_dataset/create_dataset_string_splitting.py
"""

from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import register_dataset
from ..multilingual.base_classes import MultilingualProceduralDataset
from ..config import DatasetConfig

DATASET_NAME = "string_splitting"

@dataclass
class StringSplittingConfig(DatasetConfig):
    """Configuration for String Splitting dataset generation"""

    min_initial_machines: int = 0  # Minimum number of initial machines
    max_initial_machines: int = 5  # Maximum number of initial machines
    max_iterations: int = 1_000  # Maximum number of iterations to apply the rules (Safety check for infinite loops)
    languages: list[str] | str = "en"
    language_weights: Optional[list[float]] = None

    def validate(self):
        """Validate configuration parameters"""
        assert 0 <= self.min_initial_machines, "min_initial_machines must be non-negative"
        assert (
            self.min_initial_machines <= self.max_initial_machines
        ), "min_initial_machines must be less than or equal to max_initial_machines"
        assert 0 < self.max_iterations, "max_iterations must be positive"

class StringSplittingDataset(MultilingualProceduralDataset):
    """Generates String Splitting exercises with configurable difficulty"""

    def __init__(self, config: StringSplittingConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _apply_rule(self, counts: list[int]) -> list[int]:
        """
        Apply the first applicable rule to the given counts.
        In case no rule is applicable, the counts are returned unchanged.
        """
        # label the indices for the counts
        A, B, C, X, Y, Z = range(6)

        # Rule 1: A -> 2X + Y
        if counts[A] >= 1:
            counts[A] -= 1
            counts[X] += 2
            counts[Y] += 1
        # Rule 2: 2B -> X
        elif counts[B] >= 2:
            counts[B] -= 2
            counts[X] += 1
        # Rule 3: 2C -> Y
        elif counts[C] >= 2:
            counts[C] -= 2
            counts[Y] += 1
        # Rule 4: B + C -> A
        elif counts[B] >= 1 and counts[C] >= 1:
            counts[B] -= 1
            counts[C] -= 1
            counts[A] += 1
        # Rule 5: X + Y -> Z
        elif counts[X] >= 1 and counts[Y] >= 1:
            counts[X] -= 1
            counts[Y] -= 1
            counts[Z] += 1

        return counts

    def _get_answer(self, A_machine: int, B_machine: int, C_machine: int) -> list[list[int]]:
        """Calculate the answer for a given input"""
        # counts for A B C X Y Z
        counts = [A_machine, B_machine, C_machine, 0, 0, 0]
        states = [counts]

        for _ in range(self.config.max_iterations):
            new_counts = self._apply_rule(counts[:])
            if new_counts in states:
                break
            states.append(new_counts)
            counts = new_counts

        return states

    def _generate_item(self, idx: int, language: str) -> dict[str, Any]:
        """Generate a single String Splitting question in the specified language"""
        rng = Random(self.seed + idx)

        A_machine = rng.randint(self.config.min_initial_machines, self.config.max_initial_machines)
        B_machine = rng.randint(self.config.min_initial_machines, self.config.max_initial_machines)
        C_machine = rng.randint(self.config.min_initial_machines, self.config.max_initial_machines)

        states = self._get_answer(A_machine, B_machine, C_machine)
        answer = states[-1]
        answer_str = " ".join(str(x) for x in answer)

        question = self._get_translation(
            "question_template", 
            language,
            A_machine=A_machine,
            B_machine=B_machine,
            C_machine=C_machine,
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
                "initial_machines": (A_machine, B_machine, C_machine),
                "difficulty": {
                    "initial_machines": (self.config.min_initial_machines, self.config.max_initial_machines),
                },
                "language": language,
            },
        }

class StringSplittingCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(StringSplittingCurriculum.__name__, StringSplittingConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="initial_machines",
                levels=[10, 50, 100, 500],
                description="Number of initial machines",
                lower_field_name="min_initial_machines",
                upper_field_name="max_initial_machines",
                ensure_interval=True,
            )
        )

register_dataset(DATASET_NAME, StringSplittingDataset, StringSplittingConfig, StringSplittingCurriculum)
