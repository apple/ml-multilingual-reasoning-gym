"""
Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""

"""Mahjong Puzzle Generator

https://github.com/yongchao98/CodeSteer-v1.0/blob/main/create_dataset/create_dataset_mahjong.py
"""

import string
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import register_dataset
from ..multilingual.base_classes import MultilingualProceduralDataset
from ..config import DatasetConfig


DATASET_NAME = "mahjong_puzzle"

@dataclass
class MahjongPuzzleConfig(DatasetConfig):
    """Configuration for Mahjong Puzzle dataset generation"""

    min_num_rounds: int = 10
    max_num_rounds: int = 50
    languages: list[str] | str = "en"
    language_weights: Optional[list[float]] = None

    def validate(self):
        """Validate configuration parameters"""
        assert 1 <= self.min_num_rounds, "min_num_rounds must be greater than 0"
        assert self.min_num_rounds <= self.max_num_rounds, "min_num_rounds must be less than max_num_rounds"

class MahjongPuzzleDataset(MultilingualProceduralDataset):
    """Generates Mahjong Puzzle exercises with configurable difficulty"""
    
    DATASET_NAME = "mahjong_puzzle"

    def __init__(self, config: MahjongPuzzleConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)
        self.vocabulary = list(string.ascii_uppercase)
        self.k = 13

    def _get_initial_string(self, rng: Random) -> str:
        """Generate a random string of letters"""
        pool = self.vocabulary * 2  # ensure at most 2 of each letter in initial string
        characters = rng.sample(pool, self.k)
        return "".join(characters)

    def _check_peng(self, cards: str, new_card: str) -> bool:
        """Check if a Peng pattern exists with the new card"""
        return cards.count(new_card) + 1 >= 3

    def _check_chi(self, cards: str, new_card: str) -> bool:
        """Check if a Chi pattern exists with the new card"""
        all_cards = sorted(list(cards + new_card))
        for i in range(len(all_cards) - 2):
            seq = all_cards[i : i + 3]
            if ord(seq[1]) == ord(seq[0]) + 1 and ord(seq[2]) == ord(seq[1]) + 1 and new_card in seq:
                return True
        return False

    def _simulate_game(self, rng: Random, cards: str, num_rounds: int) -> tuple[str, list]:
        """
        Simulate a game of Mahjong Puzzle

        Returns:
        - result: The final result of the game
        - rounds: List of operations (add/remove) in each round
        """

        result, rounds = None, []

        for _ in range(num_rounds):
            # Try to create interesting patterns, such as Peng or Chi
            round_outcome = rng.choice(["Peng", "Chi", "Pass"])
            if round_outcome == "Peng" and any(self._check_peng(cards, c) for c in self.vocabulary):
                new_card = rng.choice([c for c in self.vocabulary if self._check_peng(cards, c)])
                result = "Peng"
            elif round_outcome == "Chi" and any(self._check_chi(cards, c) for c in self.vocabulary):
                new_card = rng.choice([c for c in self.vocabulary if self._check_chi(cards, c)])
                result = "Chi"
            else:
                new_card = rng.choice(self.vocabulary)
                result = "Pass"

            # Update states
            remove_card = rng.choice(cards)
            cards = cards.replace(remove_card, "", 1) + new_card
            rounds.append({"add": new_card, "remove": remove_card, "cards": cards, "result": result})

        return result, rounds


    def _generate_item(self, idx: int, language: str) -> dict[str, Any]:
        """Generate a single Mahjong Puzzle question for the specified language"""
        rng = Random(self.seed + idx)

        cards = self._get_initial_string(rng)
        num_rounds = rng.randint(self.config.min_num_rounds, self.config.max_num_rounds)
        answer, rounds = self._simulate_game(rng, cards, num_rounds)
        
        # Format operations with language-specific templates
        operations_lines = []
        for i, r in enumerate(rounds):
            operation = self._get_translation(
                "round_operation", language, 
                round_num=i+1, 
                add_card=r['add'], 
                remove_card=r['remove']
            )
            operations_lines.append(operation)
        operations = "\n".join(operations_lines)
        
        # Build the question using translations
        question = self._get_translation("prompt_template", language, cards=cards, operations=operations)

        return {
            "question": question,
            "answer": answer,
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "rounds": rounds,
                "solution": answer,
                "language": language,
                "difficulty": {
                    "num_rounds": (self.config.min_num_rounds, self.config.max_num_rounds),
                },
            },
        }

class MahjongPuzzleCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(MahjongPuzzleCurriculum.__name__, MahjongPuzzleConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="num_rounds",
                levels=[10, 50, 100, 500],
                description="Number of rounds in the game",
                lower_field_name="min_num_rounds",
                upper_field_name="max_num_rounds",
            )
        )

register_dataset(DATASET_NAME, MahjongPuzzleDataset, MahjongPuzzleConfig, MahjongPuzzleCurriculum)
