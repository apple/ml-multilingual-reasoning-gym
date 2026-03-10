"""
Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""

"""Spell backward task generator"""

import re
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..data import read_data_file
from ..factory import register_dataset
from ..multilingual.base_classes import MultilingualProceduralDataset
from ..config import DatasetConfig

DATASET_NAME = "spell_backward"

@dataclass
class SpellBackwardConfig(DatasetConfig):
    """Configuration for spelling words backward task generation"""

    min_word_len: int = 3  # Minimum word length
    max_word_len: int = 10  # Maximum word length
    data_file: str = "words3to10.txt"
    languages: list[str] | str = "en"
    language_weights: Optional[list[float]] = None

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.min_word_len > 0, "min_word_len must be positive"
        assert self.max_word_len >= self.min_word_len, "max_word_len must be >= min_word_len"

class SpellBackwardDataset(MultilingualProceduralDataset):
    """Generates tasks to spell words backward"""

    def __init__(self, config: SpellBackwardConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

        # Load and preprocess text
        text = read_data_file(self.config.data_file)
        self.words = [
            word.strip()
            for word in text.splitlines()
            if word.strip().isalnum() and config.min_word_len <= len(word.strip()) <= config.max_word_len
        ]

    def _generate_item(self, idx: int, language: str) -> dict[str, Any]:
        """Generate a single spell backward task
        
        Args:
            idx: Index of the item to generate
            language: Language code for the task
            
        Returns:
            Dictionary containing question, answer, and metadata
        """
        rng = Random(self.seed + idx)

        # Select random word
        word = rng.choice(self.words)
        answer = word[::-1]

        # Generate question using translations
        question = self._get_translation("question_template", language, word=word)

        return {
            "question": question,
            "answer": answer,
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "word": word,
                "word_len": len(word),
                "language": language,
                "difficulty": {
                    "word_len": (self.config.min_word_len, self.config.max_word_len),
                },
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        reward = 0.0
        expected_answer = entry["answer"]
        if isinstance(answer, str):
            try:
                expected_answer = expected_answer.lower()
                answer = answer.lower()
                if expected_answer == answer:
                    return 1.0
                else:
                    answer_len = len(expected_answer)
                    for i in range(len(expected_answer)):
                        if i < len(expected_answer) and i < len(answer):
                            if expected_answer[i] == answer[i]:
                                reward += 1 / answer_len
                            else:
                                continue
                        else:
                            break
                    if reward == 1.0:
                        reward -= 0.2
            except:
                reward = 0.0
        return reward

class SpellBackwardCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(SpellBackwardCurriculum.__name__, SpellBackwardConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="word_len",
                levels=list(range(3, 11, 1)),
                description="Word length",
                lower_field_name="min_word_len",
                upper_field_name="max_word_len",
                ensure_interval=False,
            ),
        )

register_dataset(DATASET_NAME, SpellBackwardDataset, SpellBackwardConfig, SpellBackwardCurriculum)
