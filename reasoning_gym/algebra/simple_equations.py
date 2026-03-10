"""
Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""

import random
import string
from dataclasses import dataclass, field
from typing import Any, Optional

from sympy import Symbol

from ..coaching import BaseCurriculum, ScalarAttributeDefinition
from ..factory import register_dataset
from ..multilingual.base_classes import MultilingualProceduralDataset
from ..config import DatasetConfig

DATASET_NAME = "simple_equations"

@dataclass
class SimpleEquationsConfig(DatasetConfig):
    """Configuration for simple equation task generation"""

    min_terms: int = 2  # Minimum number of terms in expression
    max_terms: int = 4  # Maximum number of terms
    min_value: int = 1  # Minimum value for constants
    max_value: int = 100  # Maximum value for constants
    operators: tuple = ("+", "-", "*")  # Allowed operators
    operators_weights: list[float] = field(default_factory=lambda: [0.4, 0.4, 0.2])  # Weights for each operator
    languages: list[str] | str = "en"  # Add language support
    language_weights: Optional[list[float]] = None  # Add language weights support

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.min_terms > 0, "min_terms must be positive"
        assert self.max_terms >= self.min_terms, "max_terms must be >= min_terms"
        assert self.min_value > 0, "min_value must be positive"
        assert self.max_value >= self.min_value, "max_value must be >= min_value"
        assert len(self.operators) > 0, "must specify at least one operator"
        assert all(op in ("+", "-", "*") for op in self.operators), "invalid operator specified"
        assert round(sum(self.operators_weights), 1) == 1.0, "operators_weights must sum to 1.0"

class SimpleEquationsDataset(MultilingualProceduralDataset):
    """Generates simple equations with one variable to solve"""

    def __init__(self, config: SimpleEquationsConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _generate_item(self, idx: int, language: str) -> dict[str, Any]:
        """Generate a single equation task

        Returns:
            dict with keys:
                - question: str, the equation to solve (e.g. "3 * x = 12")
                - answer: str, the solution value (e.g. "4")
                - metadata: dict with generation parameters
        """
        rng = random.Random(self.seed + idx)

        # Get variable and generate equation
        variable = self._get_variable(rng)
        equation, solution = self._generate_equation(rng, variable)

        prompt_template_key = f"prompt_template_{rng.randint(1, 3)}"
        # Use translation system for prompt templates
        prompt_template = self._get_translation(prompt_template_key, language, 
                                               variable=variable, equation=equation)

        return {
            "question": prompt_template,
            "answer": str(solution),
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "equation": equation,
                "variable": variable,
                "language": language,
                "difficulty": {
                    "min_terms": self.config.min_terms,
                    "max_terms": self.config.max_terms,
                    "min_value": self.config.min_value,
                    "max_value": self.config.max_value,
                    "operators_weights": self.config.operators_weights,
                },
            },
        }

    def _get_variable(self, rng: random.Random) -> str:
        """Get a random lowercase variable name"""
        return rng.choice(string.ascii_lowercase)

    def _generate_equation(self, rng: random.Random, variable: str) -> tuple[str, int]:
        """Generate an equation and its solution

        Args:
            rng: Random number generator
            variable: Variable symbol to use in equation

        Returns:
            Tuple of (equation string, solution integer)
        """
        x = Symbol(variable)

        # Generate terms for left side
        num_terms = rng.randint(self.config.min_terms, self.config.max_terms)
        terms = []

        # Generate all constant terms first
        for _ in range(num_terms):
            value = rng.randint(self.config.min_value, self.config.max_value)
            terms.append(value)

        # Replace one random term with the variable term
        var_pos = rng.randint(0, num_terms - 1)
        coef = rng.randint(self.config.min_value, self.config.max_value)
        if "*" in self.config.operators:
            terms[var_pos] = coef * x
        else:
            terms[var_pos] = x

        # Apply operators between terms
        expr = terms[0]
        for i in range(1, num_terms):
            op = rng.choices(self.config.operators, weights=self.config.operators_weights, k=1)[0]
            if op == "+":
                expr = expr + terms[i]
            elif op == "-":
                expr = expr - terms[i]
            else:  # '*'
                expr = expr * terms[i]

        left_side = expr
        solution_value = rng.randint(self.config.min_value, self.config.max_value)
        right_side = left_side.subs(x, solution_value)
        return f"{left_side} = {right_side}", solution_value

class SimpleEquationsCurriculum(BaseCurriculum):
    """Curriculum for simple equations task"""

    def __init__(self):
        super().__init__(SimpleEquationsCurriculum.__name__, SimpleEquationsConfig)

        # Define attributes
        self._define_attributes(
            ScalarAttributeDefinition(
                name="min_terms",
                field_name="min_terms",
                levels=[2, 3, 4, 5],
                description="Minimum number of terms in simple equations",
            ),
            ScalarAttributeDefinition(
                name="max_terms",
                field_name="max_terms",
                levels=[5, 10, 15, 20],
                description="Maximum number of terms in simple equations",
            ),
            ScalarAttributeDefinition(
                name="min_value",
                field_name="min_value",
                levels=[1, 10, 100, 1000],
                description="Minimum value for constants in simple equations",
            ),
            ScalarAttributeDefinition(
                name="max_value",
                field_name="max_value",
                levels=[100, 10000, 1000000, 100000000],
                description="Maximum value for constants in simple equations",
            ),
            ScalarAttributeDefinition(
                name="operators_weights",
                field_name="operators_weights",
                levels=[[0.4, 0.4, 0.2], [0.35, 0.35, 0.3], [0.3, 0.3, 0.4], [0.2, 0.2, 0.6]],
                description="Weights for each operator in simple equations",
            ),
        )

register_dataset(DATASET_NAME, SimpleEquationsDataset, SimpleEquationsConfig, SimpleEquationsCurriculum)
