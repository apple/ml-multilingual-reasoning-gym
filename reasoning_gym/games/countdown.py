"""
Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""

import re
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

import numpy as np
import sympy
from sympy import Symbol, symbols
from sympy.parsing.sympy_parser import parse_expr

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import register_dataset
from ..multilingual.base_classes import MultilingualProceduralDataset
from ..config import DatasetConfig

DATASET_NAME = "countdown"

_num_re = re.compile(r"\b\d+\b")  # pre-compile once, reuse

def _extract_ints(expr_str: str) -> list[int]:
    """
    Fast path: grab the literal integers that appear in the source text.
    Handles duplicates correctly (e.g. “1 + 1 + 81” ⇒ [1, 1, 81]).
    """
    return [int(m) for m in _num_re.findall(expr_str)]

@dataclass
class CountdownConfig(DatasetConfig):
    """Configuration for Countdown Number Game task generation"""

    min_numbers: int = 4  # Minimum numbers to provide
    max_numbers: int = 6  # Maximum numbers to provide
    min_value: int = 1  # Minimum value for source numbers
    max_value: int = 100  # Maximum value for source numbers
    min_target: int = 100  # Minimum target value
    max_target: int = 999  # Maximum target value
    operators: tuple = ("+", "-", "*", "/")  # Allowed operators
    shuffle: bool = True  # Whether to shuffle the order of source numbers
    languages: list[str] | str = "en"  # Language support
    language_weights: Optional[list[float]] = None  # Language weights

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.min_numbers > 1, "min_numbers must be greater than 1"
        assert self.max_numbers >= self.min_numbers, "max_numbers must be >= min_numbers"
        assert self.min_value > 0, "min_value must be positive"
        assert self.max_value >= self.min_value, "max_value must be >= min_value"
        assert self.min_target > 0, "min_target must be positive"
        assert self.max_target >= self.min_target, "max_target must be >= min_target"
        assert len(self.operators) > 0, "must specify at least one operator"
        assert all(op in ("+", "-", "*", "/") for op in self.operators), "invalid operator specified"

class CountdownDataset(MultilingualProceduralDataset):
    """Generates Countdown Number Game tasks"""

    def __init__(self, config: CountdownConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _generate_item(self, idx: int, language: str) -> dict:
        """Generate a single Countdown Game task

        Args:
            idx: Index of the item to generate
            language: Language for the task

        Returns:
            dict with keys:
                - question: str, the task description with numbers and target
                - answer: str, one possible solution expression
                - metadata: dict with generation parameters
        """
        rng = Random(self.seed + idx)

        # Generate a valid expression and its result
        expression, numbers, target = self._generate_expression(rng)

        # Optionally randomize the order of numbers
        if self.config.shuffle:
            rng.shuffle(numbers)

        list_separator = self._get_translation("list_separator", language)
        numbers_str = list_separator.join(map(str, numbers))

        # Use translation system for the question
        choice = rng.randint(0, 2)
        question_template = self._get_translation(f"question_template_{choice}", language, numbers=numbers_str, target=target)

        # Use translation system for the format template
        format_template = self._get_translation("format_template", language, question=question_template)

        return {
            "question": format_template,
            "answer": expression,
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "numbers": numbers,
                "target": target,
                "expression": expression,
                "language": language,
                "difficulty": {
                    "numbers": (self.config.min_numbers, self.config.max_numbers),
                    "target": (self.config.min_target, self.config.max_target),
                    "value": (self.config.min_value, self.config.max_value),
                },
            },
        }

    def _generate_candidate_expression(self, rng: Random, num_terms: int) -> tuple[sympy.Expr, list[int], list[Symbol]]:
        """Generate a candidate expression with random numbers and operators

        Args:
            rng: Random number generator
            num_terms: Number of terms to include

        Returns:
            Tuple of (sympy expression, list of numbers, list of symbols)
        """
        # Generate random numbers
        numbers = [rng.randint(self.config.min_value, self.config.max_value) for _ in range(num_terms)]

        # Create symbols for building expression
        syms = symbols(f"x:{num_terms}")

        # Build random expression
        expr = syms[0]

        for i in range(1, num_terms):
            op = rng.choice(self.config.operators)
            if op == "+":
                expr = expr + syms[i]
            elif op == "-":
                expr = expr - syms[i]
            elif op == "*":
                expr = expr * syms[i]
            else:  # division
                # Handle division carefully to ensure integer results
                if numbers[i] != 0:  # Avoid division by zero
                    # Get current value after substituting previous numbers
                    current = int(expr.subs({sym: num for sym, num in zip(syms[:i], numbers[:i])}))
                    # Try each remaining number to find one that divides evenly
                    remaining = [n for n in numbers[i:] if n != 0]
                    rng.shuffle(remaining)  # Randomize order for variety
                    found_divisor = False
                    for div in remaining:
                        if current % div == 0:  # Check if divides evenly
                            numbers[i] = div
                            expr = expr / syms[i]
                            found_divisor = True
                            break
                    if not found_divisor:
                        # If no number divides evenly, fallback to subtraction
                        expr = expr - syms[i]
                else:
                    # Fallback to addition for zero
                    expr = expr + syms[i]

        return expr, numbers, syms

    def _generate_expression(self, rng: Random) -> tuple[str, list[int], int]:
        """Generate a valid expression and its result

        Returns:
            Tuple of (expression string, list of numbers used, target value)
        """
        num_terms = rng.randint(self.config.min_numbers, self.config.max_numbers)

        max_attempts = 200
        for attempt in range(max_attempts):
            try:
                expr, numbers, syms = self._generate_candidate_expression(rng, num_terms)

                # Substitute actual numbers to get target
                subs = {sym: num for sym, num in zip(syms, numbers)}
                target = int(expr.subs(subs))

                # Convert to string expression
                expr_str = str(expr)
                for i, sym in enumerate(syms):
                    expr_str = expr_str.replace(str(sym), str(numbers[i]))

                # Ensure target is within bounds
                if self.config.min_target <= target <= self.config.max_target:
                    return expr_str, numbers, target

            except (ValueError, ZeroDivisionError):
                continue
        
        print(f"Countdown: Failed to generate expression satisfying the curriculum constraints after {max_attempts} attempts",  expr_str, numbers, target)
        # return valid countdown expression even if it doesn't necessarily conform to the curriculum
        # raise ValueError(f"Failed to generate valid expression after {max_attempts} attempts")
        return expr_str, numbers, target

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Determine if the solution provided solves the problem"""
        reward = 0.01  # Default reward

        if answer is None or not answer.strip():
            return reward

        try:
            user_answer = float(parse_expr(answer))
            used_numbers = _extract_ints(answer)
            target_numbers = entry["metadata"]["numbers"]

            if sorted(used_numbers) != sorted(target_numbers):
                return 0.05

            if np.isclose(user_answer, entry["metadata"]["target"], atol=1e-6):
                return 1.0

            return 0.05 if answer else 0.01
        except Exception:
            return 0.01

class CountdownCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(CountdownCurriculum.__name__, CountdownConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="numbers",
                levels=[3, 6, 9, 12, 15],
                description="Number of source numbers",
                lower_field_name="min_numbers",
                upper_field_name="max_numbers",
                ensure_interval=True,
            ),
            RangeAttributeDefinition(
                name="target",
                levels=[100, 500, 1000, 5000, 10000],
                description="Target number to reach",
                lower_field_name="min_target",
                upper_field_name="max_target",
                ensure_interval=True,
            ),
            RangeAttributeDefinition(
                name="value",
                levels=[1, 100, 200, 300],
                description="Value of numbers",
                lower_field_name="min_value",
                upper_field_name="max_value",
                ensure_interval=True,
            ),
        )

# Register the dataset
register_dataset(DATASET_NAME, CountdownDataset, CountdownConfig, CountdownCurriculum)
