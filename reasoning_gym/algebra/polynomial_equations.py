"""
Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""

import math
import random
from dataclasses import dataclass
from typing import Any, Optional

from sympy import Eq, Symbol, expand, solve

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import register_dataset
from ..config import DatasetConfig
from ..multilingual.base_classes import MultilingualProceduralDataset

DATASET_NAME = "polynomial_equations"

@dataclass
class PolynomialEquationsConfig(DatasetConfig):
    """
    Configuration for polynomial equation task generation.
    """

    min_terms: int = 2  # Minimum number of polynomial terms
    max_terms: int = 4  # Maximum number of polynomial terms
    min_value: int = 1  # Minimum value for coefficients
    max_value: int = 100  # Maximum value for coefficients
    min_degree: int = 1  # Minimum polynomial degree
    max_degree: int = 3  # Maximum polynomial degree
    operators: tuple[str, ...] = (
        "+",
        "-",
    )  # Allowed operators between terms, Avoid adding '*' or '/' because they will affect the degree
    # reward function hyperparameters
    penalty_missing_factor = 0.5
    penalty_extra_factor = 0.5
    exp_distance_factor = -10.0
    languages: list[str] | str = "en"
    language_weights: Optional[list[float]] = None

    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.min_terms > 0, "min_terms must be positive."
        assert self.max_terms >= self.min_terms, "max_terms must be >= min_terms."

        assert self.min_value > 0, "min_value must be positive."
        assert self.max_value >= self.min_value, "max_value must be >= min_value."

        assert self.min_degree >= 1, "min_degree must be >= 1."
        assert self.max_degree >= self.min_degree, "max_degree must be >= min_degree."

        allowed_ops = {"+", "-"}
        assert len(self.operators) > 0, "operators tuple cannot be empty."
        assert all(op in allowed_ops for op in self.operators), "Invalid operator found. Must be a subset of {+, -}."

class PolynomialEquationsDataset(MultilingualProceduralDataset):
    """
    Generates random polynomial equations of degree in [min_degree, max_degree].
    - The polynomial is formed by summing random terms of the form: coeff * x^exponent.
    - Then we solve "polynomial_expr = 0" using Sympy.
    - The solution may be real or complex; we filter real solutions by default for simplicity.
    """

    def __init__(self, config: PolynomialEquationsConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _generate_item(self, idx: int, language: str) -> dict[str, Any]:
        """
        Generate a single polynomial equation item.

        Args:
            idx: Index of the item to generate
            language: Language code for the item

        Returns:
            A dict with:
                - question: str (e.g. "Solve the polynomial equation: 2*x**2 - 3*x + 1 = 0")
                - answer: str (the sorted list of real solutions, e.g. "0.5, 1.0")
                - metadata: dict with details (polynomial_expr, degree, etc.)
        """
        rng = random.Random(self.seed + idx)
        for _ in range(
            20
        ):  # Increase the number of attempts to get a solvable polynomial - many solutions only real solution is 0
            # Get variable and generate polynomial equation in standard form
            variable = self._get_variable(rng)
            degree = rng.randint(self.config.min_degree, self.config.max_degree)
            num_terms = rng.randint(self.config.min_terms, self.config.max_terms)
            polynomial_expr = self._generate_polynomial_expr(rng, variable, degree, num_terms)
            polynomial_expanded = expand(polynomial_expr)

            # Solve the polynomial = 0
            # We filter real solutions only
            solutions = solve(Eq(polynomial_expanded, 0), variable, dict=False)
            real_solutions = []
            for sol in solutions:
                if sol.is_real:
                    # Evaluate symbolic solution to a floating approximation
                    real_solutions.append(round(float(sol.evalf()), 4))

            if len(real_solutions) > 0:
                real_solutions.sort()
                break

        answer_str = ", ".join(str(x) for x in real_solutions)
        
        # Select random prompt template and format question
        prompt_template_key = f"prompt_template_{rng.randint(1, 4)}"
        question_text = self._get_translation(prompt_template_key, language, 
                                            variable=variable, 
                                            polynomial_expanded=polynomial_expanded)
        
        instruction_text = self._get_translation("instruction_text", language)
        question = question_text + instruction_text

        return {
            "question": question,
            "answer": answer_str,
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "polynomial_expr": str(polynomial_expanded),
                "variable": variable,
                "degree": degree,
                "real_solutions": real_solutions,
                "num_terms": num_terms,
                "language": language,
                "difficulty": {
                    "terms": (self.config.min_terms, self.config.max_terms),
                    "degree": (self.config.min_degree, self.config.max_degree),
                },
            },
        }

    def _get_variable(self, rng: random.Random) -> str:
        """Get a random lowercase variable name"""
        return rng.choice("abcdefghklmnopqrstuvwxyz")  # remove ij to avoid confusion with complex numbers

    def _generate_polynomial_expr(self, rng: random.Random, variable: Symbol, degree: int, num_terms: int):
        """
        Randomly generate a polynomial expression of 'degree'.
        We'll use the config parameters:
            - min_terms, max_terms: how many total terms to combine
            - min_value, max_value: range for coefficients
            - operators: to decide sign flips or direct addition

         Args:
            rng: Random number generator
            variable: Variable symbol to use in equation
            degree: Highest degree. We ensure that there is at least one term with exponent=degree

        Returns:
            Polynomial string
        """
        x = Symbol(variable)

        # Choose the number of terms and their respective degrees
        # Keep track of exponents, exponents can repeat or skip but we force the highest exponent
        chosen_exponents = [degree]
        # Fill the rest randomly in [0, degree]
        for _ in range(num_terms - 1):
            exp = rng.randint(0, degree)
            chosen_exponents.append(exp)

        # Now build the polynomial expression: sum_{term}( coeff * x^exponent ), with optional sign
        polynomial_expr = 0
        for exp in chosen_exponents:
            coeff = rng.randint(self.config.min_value, self.config.max_value)
            # If '-' in operators, we can randomly flip the sign
            if "-" in self.config.operators and rng.random() < 0.5:
                coeff = -coeff
            term_expr = coeff * (x**exp)
            polynomial_expr += term_expr

        return polynomial_expr

    def _parse_score_to_list(self, answer: Optional[str]) -> list[float]:
        """Parses a comma-separated string of scores into a sorted list of floats.

        This method takes a string containing comma-separated numeric values,
        attempts to convert each value to a float, and returns a sorted list of these floats.
        Any values that cannot be converted to a float are ignored.
        Handles empty strings gracefully.

        Args:
            answer: An optional string containing comma-separated numeric values.
            Can be None or an empty string.
        Returns:
            A sorted list of floats parsed from the input string.
            Returns an empty list if the input is None, empty, or contains no valid numeric values.
        """

        if answer is None or len(answer) == 0:  # Handle None or empty input
            return []

        output_float_vals = []
        for output_val in answer.split(","):
            try:
                # Convert to float, strip whitespace
                output_float_vals.append(float(output_val.strip()))
            except ValueError:
                # Ignore values that cannot be converted to float
                continue

        return sorted(output_float_vals)  # Return the sorted list of floats

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """
        Score an answer based on its numerical distance to oracle solutions using exponential decay.
        This function compares a predicted answer (or list of answers) to a set of oracle solutions
        (also a list of numbers). It calculates a reward based on how close the predicted solutions
        are to the oracle solutions, using an exponential decay function.  It also applies penalties
        for missing or extra predicted solutions. The implementation is a greedy algorithm where we
        find the closest matching oracle solution for a given predicted solution and only allow an
        oracle solution to match once.

        Args:
            answer: The predicted answer (or a string that can be parsed into a list of numbers).
                    May be None.
            entry: A dictionary containing the oracle solution(s) under the key "answer"
                (which can be a string that can be parsed into a list of numbers).

        Returns:
            A float representing the final score. The score is non-negative.
        """
        oracle_solutions = self._parse_score_to_list(entry["answer"])  # Parse oracle solutions
        predicted_solutions = self._parse_score_to_list(answer)  # Parse predicted solutions

        if len(oracle_solutions) == 0 and len(predicted_solutions) == 0:
            return 1.0

        total_reward = 0.0
        matched_solutions = 0
        extra_solutions = 0
        missing_solutions = 0

        for predicted_solution in predicted_solutions:

            # find the closest matching solution from the oracle solutions.
            # this is a greedy approach to computing the score
            matched_distance = float("inf")
            matched_distance_index = None
            for oracle_solution_index, oracle_solution in enumerate(oracle_solutions):
                if matched_distance > abs(predicted_solution - oracle_solution):
                    matched_distance = abs(predicted_solution - oracle_solution)
                    matched_distance_index = oracle_solution_index

            if matched_distance_index is not None:
                matched_solutions += 1
                # Remove matched oracle solution
                oracle_solutions.pop(matched_distance_index)
                # Exponential decay reward
                total_reward += math.exp(matched_distance * self.config.exp_distance_factor)
            else:
                # Extra predicted solution
                extra_solutions += 1

        # Count remaining oracle solutions as missing
        for oracle_solution in oracle_solutions:
            missing_solutions += 1

        # Calculate penalty for either missing or extra solutions
        penalty = missing_solutions * self.config.penalty_missing_factor
        penalty += extra_solutions * self.config.penalty_extra_factor

        if matched_solutions > 0:
            # normalize the rewards that we found matching solutions for
            # so that the value is bounded between 0 and 1
            total_reward = total_reward / matched_solutions

        # Final reward capped at 0
        final_reward = max(0, total_reward - penalty)

        return final_reward

class PolynomialEquationsCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(PolynomialEquationsCurriculum.__name__, PolynomialEquationsConfig)
        self._define_attributes(
            RangeAttributeDefinition(
                name="degree",
                levels=[1, 2, 3, 4],
                lower_field_name="min_degree",
                upper_field_name="max_degree",
                description="The degree of the polynomial equation",
                ensure_interval=True,
            ),
            RangeAttributeDefinition(
                name="terms",
                levels=[2, 3, 4, 5],
                lower_field_name="min_terms",
                upper_field_name="max_terms",
                description="The number of terms in the polynomial equation",
            ),
        )

register_dataset(DATASET_NAME, PolynomialEquationsDataset, PolynomialEquationsConfig, PolynomialEquationsCurriculum)
