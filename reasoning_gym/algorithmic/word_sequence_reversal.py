"""
Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""

"""Word reversal task generator"""

import re
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..data import read_data_file
from ..factory import register_dataset
from ..multilingual.base_classes import MultilingualProceduralDataset
from ..config import DatasetConfig

# Question template moved to translation files

DATASET_NAME = "word_sequence_reversal"

@dataclass
class WordSequenceReversalConfig(DatasetConfig):
    """Configuration for word sequence reversal task generation"""

    min_words: int = 3  # Minimum words in list
    max_words: int = 8  # Maximum words in list
    languages: list[str] | str = "en"  # Supported languages
    language_weights: Optional[list[float]] = None  # Optional language weights

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.min_words > 0, "min_words must be positive"
        assert self.max_words >= self.min_words, "max_words must be >= min_words"

class WordSequenceReversalDataset(MultilingualProceduralDataset):
    """Generates word sequence reversal tasks from text spans"""

    def __init__(self, config: WordSequenceReversalConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

        # Load and preprocess text
        text = read_data_file("in_the_year_2889.txt")
        # Extract words and clean them to contain only alphanumeric characters
        self.words = [word for word in re.findall(r"\b\w+\b", text) if word.isalnum()]

    def _generate_item(self, idx: int, language: str) -> dict[str, Any]:
        """Generate a single word reversal task for the given index and language"""
        rng = Random(self.seed + idx)

        # Select random words
        num_words = min(
            rng.randint(self.config.min_words, self.config.max_words),
            len(self.words),
        )
        word_indices = rng.sample(range(len(self.words)), num_words)
        words = [self.words[i] for i in word_indices]

        # Create question and answer
        words_str = ", ".join(words)
        answer = ", ".join(reversed(words))

        # Create question using translation
        question = self._get_translation("question_template", language, words=words_str)
        
        return {
            "question": question,
            "answer": answer,
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "num_words": num_words,
                "words": words,
                "language": language,
                "difficulty": {
                    "words": (self.config.min_words, self.config.max_words),
                },
            },
        }

class WordSequenceReversalCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(WordSequenceReversalCurriculum.__name__, WordSequenceReversalConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="words",
                levels=[10, 25, 50, 100],
                description="Number of words in the list",
                lower_field_name="min_words",
                upper_field_name="max_words",
                ensure_interval=True,
            ),
        )

register_dataset(DATASET_NAME, WordSequenceReversalDataset, WordSequenceReversalConfig, WordSequenceReversalCurriculum)
