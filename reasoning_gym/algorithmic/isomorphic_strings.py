"""
Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""

"""Check if two strings are isomorphic.

Two strings are isomorphic if the characters in one string can be replaced to get the second string.

A popular Leetcode problem:
https://leetcode.com/problems/isomorphic-strings/description/
"""

from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import register_dataset
from ..multilingual.base_classes import MultilingualProceduralDataset
from ..config import DatasetConfig


DATASET_NAME = "isomorphic_strings"

@dataclass
class IsomorphicStringsConfig(DatasetConfig):
    """Configuration for Isomorphic Strings dataset generation"""

    min_string_length: int = 2  # Minimum length of the strings
    max_string_length: int = 10  # Maximum length of the strings
    p_solvable: float = 0.5  # Probability that the generated question is solvable
    languages: list[str] | str = "en"  # Add this field
    language_weights: Optional[list[float]] = None  # Add this optional field

    def validate(self):
        """Validate configuration parameters"""
        assert (
            2 <= self.min_string_length <= self.max_string_length
        ), "min_string_length must be between 2 and max_string_length"
        assert 0 <= self.p_solvable <= 1, "p_solvable must be between 0 and 1"

class IsomorphicStringsDataset(MultilingualProceduralDataset):
    """Generates Isomorphic Strings exercises with configurable difficulty"""

    def __init__(self, config: IsomorphicStringsConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)
        self.letters = {chr(i) for i in range(ord("a"), ord("z") + 1)}

    def _check_isomorphic(self, s: str, t: str) -> bool:
        """Check if two strings are isomorphic"""
        if len(s) != len(t):
            return False

        mapping, inverse_mapping = {}, {}  # s -> t, t -> s
        for i in range(len(s)):
            if (s[i] in mapping and mapping[s[i]] != t[i]) or (
                t[i] in inverse_mapping and s[i] != inverse_mapping[t[i]]
            ):
                return False
            mapping[s[i]] = t[i]
            inverse_mapping[t[i]] = s[i]

        return True

    def _generate_inputs(self, rng: Random, string_length: int, solvable: bool) -> tuple[str, str]:
        """Generate the two input strings"""
        s, t = [], []
        mapping = {}

        # Generate a valid isomorphic pair first (leave one character for potential conflict)
        for _ in range(string_length - 1):
            char_s = rng.choice(sorted(self.letters))
            if char_s not in mapping:
                # Choose a random character that is not already mapped
                char_t = rng.choice(sorted(self.letters - set(mapping.values())))
                mapping[char_s] = char_t
            else:
                # Use the existing mapping
                char_t = mapping[char_s]
            s.append(char_s)
            t.append(char_t)

        if not solvable:
            # Solution should be unsolvable, create conflict
            letter = rng.choice(sorted(mapping.keys()))
            conflict = rng.choice(sorted(self.letters - {mapping[letter]}))
            insert_idx = rng.randint(0, len(s))
            s.insert(insert_idx, letter)
            t.insert(insert_idx, conflict)

        return "".join(s), "".join(t)

    def _generate_item(self, idx: int, language: str) -> dict[str, Any]:
        """Generate a single Isomorphic Strings question in the specified language"""
        rng = Random(self.seed + idx)

        string_length = rng.randint(self.config.min_string_length, self.config.max_string_length)
        solvable = rng.random() < self.config.p_solvable
        s, t = self._generate_inputs(rng, string_length, solvable)
        answer = self._check_isomorphic(s, t)

        # Build question using translations
        true_answer = self._get_translation("true_answer", language)
        false_answer = self._get_translation("false_answer", language)
        question = self._get_translation("prompt_template", language, 
                                       strings=f"{s} {t}", 
                                       true_answer=true_answer, 
                                       false_answer=false_answer)
        
        # Use localized True/False for answer
        localized_answer = true_answer if answer else false_answer

        return {
            "question": question,
            "answer": localized_answer,
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "words": [s, t],
                "solution": answer,
                "solvable": solvable,
                "string_length": string_length,
                "language": language,
                "difficulty": {
                    "string_length": (self.config.min_string_length, self.config.max_string_length),
                },
            },
        }

    def score_answer(self, answer: str, entry: dict) -> float:
        """Score the answer, handling localized True/False responses"""
        # Check if the answer matches the expected localized answer (case-insensitive)
        expected_answer = entry["answer"]
        if answer.strip().lower() == expected_answer.lower():
            return 1.0
            
        return 0.0

class IsomorphicStringsCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(IsomorphicStringsCurriculum.__name__, IsomorphicStringsConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="string_length",
                levels=[10, 50, 100, 1000],
                description="Length of the strings",
                lower_field_name="min_string_length",
                upper_field_name="max_string_length",
            )
        )

register_dataset(DATASET_NAME, IsomorphicStringsDataset, IsomorphicStringsConfig, IsomorphicStringsCurriculum)
