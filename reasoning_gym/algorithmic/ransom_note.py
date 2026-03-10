"""
Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""

"""Check if you can construct a ransom note from letters in a magazine.

A popular Leetcode problem:
https://leetcode.com/problems/ransom-note/description/
"""

from collections import defaultdict
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import register_dataset
from ..multilingual.base_classes import MultilingualProceduralDataset
from ..config import DatasetConfig

DATASET_NAME = "ransom_note"

@dataclass
class RansomNoteConfig(DatasetConfig):
    """Configuration for Ransom Note dataset generation"""

    min_note_length: int = 1  # Minimum length of the ransom note
    max_note_length: int = 10  # Maximum length of the ransom note
    min_magazine_length: int = 2  # Minimum length of the magazine
    max_magazine_length: int = 30  # Maximum length of the magazine
    p_solvable: float = 0.5  # Probability that the ransom note can be constructed
    languages: list[str] | str = "en"  # Language(s) for the dataset
    language_weights: Optional[list[float]] = None  # Optional weights for language selection

    def validate(self):
        """Validate configuration parameters"""
        # assert 1 <= self.max_note_length <= MAX_NOTE_LENGTH, "max_note_length must be between 1 and MAX_NOTE_LENGTH"
        assert 1 <= self.min_note_length, "min_note_length must be at least 1"
        assert (
            self.min_note_length <= self.max_note_length
        ), "min_note_length must be less than or equal to max_note_length"
        assert 2 <= self.min_magazine_length, "min_magazine_length must be at least 2"
        assert (
            self.min_magazine_length <= self.max_magazine_length
        ), "min_magazine_length must be less than or equal to max_magazine_length"
        assert self.max_note_length < self.max_magazine_length, "max_note_length must be less than max_magazine_length"
        assert 0 <= self.p_solvable <= 1, "p_solvable must be between 0 and 1"

class RansomNoteDataset(MultilingualProceduralDataset):
    """Generates Ransom Note exercises with configurable difficulty"""

    def __init__(self, config: RansomNoteConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)
        self.letters = {chr(i) for i in range(ord("a"), ord("z") + 1)}

    def _get_inputs(self, rng: Random, note_length: int, magazine_length: int, solvable: bool) -> tuple[str, str]:
        """Generate random ransom note and magazine"""
        ransom_note = [rng.choice(sorted(self.letters)) for _ in range(note_length)]
        magazine = ransom_note.copy()
        if solvable:
            magazine.extend([rng.choice(sorted(self.letters)) for _ in range(magazine_length - note_length)])
        else:
            remove_letter = rng.choice(magazine)
            magazine.remove(remove_letter)
            magazine.extend(
                [rng.choice(sorted(self.letters - {remove_letter})) for _ in range(magazine_length - note_length + 1)]
            )
        rng.shuffle(ransom_note)
        rng.shuffle(magazine)
        return "".join(ransom_note), "".join(magazine)

    def _can_construct(self, ransom_note: str, magazine: str) -> bool:
        """Check if ransom note can be constructed from magazine"""
        count = defaultdict(int)
        for c in magazine:
            count[c] += 1
        for c in ransom_note:
            if count[c] <= 0:
                return False
            count[c] -= 1
        return True

    def _generate_item(self, idx: int, language: str) -> dict[str, Any]:
        """Generate a single Ransom Note question with multilingual support"""
        rng = Random(self.seed + idx)

        note_length = rng.randint(self.config.min_note_length, self.config.max_note_length)
        magazine_length = rng.randint(
            max(note_length, self.config.min_magazine_length), self.config.max_magazine_length
        )
        solvable = rng.random() < self.config.p_solvable
        ransom_note, magazine = self._get_inputs(rng, note_length, magazine_length, solvable)
        answer = self._can_construct(ransom_note, magazine)

        # Use translation for the question template
        true_answer = self._get_translation("true_answer", language)
        false_answer = self._get_translation("false_answer", language)
        question = self._get_translation("question_template", language, 
                                       ransom_note=ransom_note, 
                                       magazine=magazine,
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
                "language": language,
                "ransom_note": ransom_note,
                "magazine": magazine,
                "solution": answer,
                "solvable": solvable,
                "note_length": note_length,
                "magazine_length": magazine_length,
                "difficulty": {
                    "note_length": (self.config.min_note_length, self.config.max_note_length),
                    "magazine_length": (self.config.min_magazine_length, self.config.max_magazine_length),
                },
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict) -> float:
        """Score the answer, handling localized True/False responses"""
        # Handle None or non-string answers
        if answer is None:
            return 0.0
        
        # Check if the answer matches the expected localized answer (case-insensitive)
        expected_answer = entry["answer"]
        if answer.strip().lower() == expected_answer.lower():
            return 1.0
            
        return 0.0

class RansomNoteCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(RansomNoteCurriculum.__name__, RansomNoteConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="note_length",
                levels=[10, 50, 100, 500],
                description="Length of the ransom note",
                lower_field_name="min_note_length",
                upper_field_name="max_note_length",
                ensure_interval=True,
            ),
            RangeAttributeDefinition(
                name="magazine_length",
                levels=[50, 100, 500, 1000],
                description="Length of the magazine",
                lower_field_name="min_magazine_length",
                upper_field_name="max_magazine_length",
                ensure_interval=True,
            ),
        )

register_dataset(DATASET_NAME, RansomNoteDataset, RansomNoteConfig, RansomNoteCurriculum)
