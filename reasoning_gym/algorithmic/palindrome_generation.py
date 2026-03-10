"""
Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""

import random
import string
from dataclasses import dataclass
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import register_dataset
from ..multilingual.base_classes import MultilingualProceduralDataset
from ..config import DatasetConfig

DATASET_NAME = "palindrome_generation"

@dataclass
class PalindromeConfig(DatasetConfig):
    """
    Configuration for the palindrome task.

    - min_length: Minimum length of the palindrome.
    - max_length: Maximum length of the palindrome.
    - seed: Optional seed for reproducibility.
    - size: Number of palindrome samples in the virtual dataset.
    - languages: List of languages to support or single language string.
    - language_weights: Optional weights for language sampling.
    """

    min_length: int = 3
    max_length: int = 10
    languages: list[str] | str = "en"
    language_weights: Optional[list[float]] = None

    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.min_length >= 1, "min_length must be >= 1"
        assert self.max_length >= self.min_length, "max_length must be >= min_length"

class PalindromeDataset(MultilingualProceduralDataset):
    """
    Generates a set of letters that can be assembled into a palindrome.
    """

    def __init__(self, config: PalindromeConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _generate_item(self, idx: int, language: str) -> dict[str, Any]:
        """
        Generate a single palindrome task for a specific language.

        Args:
            idx: Index of the item to generate.
            language: Language code for the generated item.

        Returns:
            dict with:
            - "question": Set of letters to form a palindrome.
            - "answer": A correct palindrome.
            - "metadata": Includes letter set and generated palindrome.
        """
        rng = random.Random(self.seed + idx)
        length = rng.randint(self.config.min_length, self.config.max_length)
        letters = self._generate_palindrome_letters(rng, length)
        scrambled_letters = rng.sample(letters, len(letters))  # Scramble the order
        palindrome = self._assemble_palindrome(letters)
        
        # Get localized letters separator
        letters_separator = self._get_translation("letters_separator", language)
        
        # Format letters string with localized separator
        letters_str = letters_separator.join(scrambled_letters)
        
        question = self._get_translation("question_template", language, letters=letters_str)
        
        return {
            "question": question,
            "answer": palindrome,
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "letters": scrambled_letters,
                "generated_palindrome": palindrome,
                "length": length,
                "language": language,
                "difficulty": {
                    "length": (self.config.min_length, self.config.max_length),
                },
            },
        }

    def _generate_palindrome_letters(self, rng: random.Random, length: int) -> list[str]:
        """Generate a set of letters that can form a palindrome."""
        half_length = length // 2
        letters = rng.choices(string.ascii_lowercase, k=half_length)
        if length % 2 == 1:
            middle_letter = rng.choice(string.ascii_lowercase)
            return letters + [middle_letter] + letters[::-1]
        return letters + letters[::-1]

    def _assemble_palindrome(self, letters: list[str]) -> str:
        """Return the palindrome string from the letter set."""
        return "".join(letters)

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Determine if the solution provided is a valid palindrome.
        The answer is expected to be a single string

        Expected behavior:
        - Correct answer (palindrome with only correct letters in the correct quantities) gives 1.0
        - An answer that is a palindrome, but not with the same letters as provided, gives 0.05
        - An answer that is a string, but not a palindrome gives 0.02
        - An empty string gives 0.0
        - None gives 0.0.
        """
        if answer is None or not isinstance(answer, str):
            return 0.0  # No answer given

        if answer == "":
            return 0.0

        metadata = entry["metadata"]
        answer = answer.strip().lower()
        expected_letters = metadata["letters"]

        # Check if the answer is a palindrome
        if answer != answer[::-1]:
            return 0.02

        # Check if answer contains the same letters as provided (ignoring order)
        if sorted(answer) != sorted(expected_letters):
            return 0.05

        return 1.0  # Correct solution

class PalindromeCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(PalindromeCurriculum.__name__, PalindromeConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="length",
                levels=[10, 50, 100, 500],
                description="Length of the generated palindrome.",
                lower_field_name="min_length",
                upper_field_name="max_length",
                ensure_interval=True,
            )
        )

register_dataset(DATASET_NAME, PalindromeDataset, PalindromeConfig, PalindromeCurriculum)
