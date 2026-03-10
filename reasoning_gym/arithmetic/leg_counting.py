"""
Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""

"""Leg counting task generator"""

from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import register_dataset
from ..multilingual.base_classes import MultilingualProceduralDataset
from ..config import DatasetConfig

ANIMALS = {
    # Animals with 0 legs
    "snake": 0,
    "sea slug": 0,
    "jellyfish": 0,
    "flatworm": 0,
    "leech": 0,
    # Animals with 2 legs
    "chicken": 2,
    "bird": 2,
    "human": 2,
    "duck": 2,
    # Animals with 4 legs
    "dog": 4,
    "cat": 4,
    "cow": 4,
    "horse": 4,
    "lion": 4,
    "elephant": 4,
    "giraffe": 4,
    "tiger": 4,
    "deer": 4,
    "sheep": 4,
    # Animals with 5 legs
    "starfish": 5,
    # Animals with 6 legs
    "insect": 6,
    "ant": 6,
    "butterfly": 6,
    "beetle": 6,
    "bee": 6,
    "wasp": 6,
    "grasshopper": 6,
    "cricket": 6,
    "cockroach": 6,
    "praying mantis": 6,
    "firefly": 6,
    # Animals with 8 legs
    "spider": 8,
    "scorpion": 8,
    # Animals with 10 legs
    "crab": 10,
    "lobster": 10,
    "shrimp": 10,
    # Animals with 14 legs
    "woodlouse": 14,
}

DATASET_NAME = "leg_counting"

@dataclass
class LegCountingConfig(DatasetConfig):
    """Configuration for leg counting task generation"""

    min_animals: int = 3  # Minimum number of animals in problem
    max_animals: int = 10  # Maximum number of animals
    min_instances: int = 1  # Minimum instances of each animal
    max_instances: int = 15  # Maximum instances of each animal
    languages: list[str] | str = "en"  # Supported languages
    language_weights: Optional[list[float]] = None  # Language weights

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.min_animals > 0, "min_animals must be positive"
        assert self.max_animals >= self.min_animals, "max_animals must be >= min_animals"
        assert self.max_animals <= len(ANIMALS), "max_animals must be <= number of available animals"  # 37
        assert self.min_instances > 0, "min_instances must be positive"
        assert self.max_instances >= self.min_instances, "max_instances must be >= min_instances"

class LegCountingDataset(MultilingualProceduralDataset):
    """Generates leg counting arithmetic tasks"""

    def __init__(self, config: LegCountingConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _generate_animals(self, rng: Random) -> dict[str, int]:
        """Generate a random set of animals and their counts"""
        num_types = rng.randint(self.config.min_animals, self.config.max_animals)
        animals = {}

        # Select random animals
        selected_animals = rng.sample(list(ANIMALS.keys()), num_types)
        for animal in selected_animals:
            count = rng.randint(self.config.min_instances, self.config.max_instances)
            animals[animal] = count

        return animals

    def _generate_item(self, idx: int, language: str) -> dict[str, Any]:
        """Generate a single leg counting task"""
        rng = Random(self.seed + idx)

        # Generate random animals and their counts
        animals = self._generate_animals(rng)

        # Calculate total legs
        total_legs = sum(count * ANIMALS[animal] for animal, count in animals.items())

        # Format animal counts using the new structured format
        animal_lines = []
        for animal, count in animals.items():
            # Use animal name directly (no plural/singular complexity)
            animal_text = self._get_translation(animal, language)
            
            # Create structured line format
            formatted_line = self._get_translation("animal_line_format", language, 
                                                 animal=animal_text, count=count)
            animal_lines.append(formatted_line)

        # Join lines with newlines
        animals_text = "\n".join(animal_lines)

        # Get question template
        question = self._get_translation("question_template", language, 
                                       animals=animals_text)

        return {
            "question": question,
            "answer": str(total_legs),
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "animals": animals,
                "num_animals": len(animals),
                "total_legs": total_legs,
                "language": language,
                "difficulty": {
                    "num_animals": (self.config.min_animals, self.config.max_animals),
                    "num_instances": (self.config.min_instances, self.config.max_instances),
                },
            },
        }

class LegCountingCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(LegCountingCurriculum.__name__, LegCountingConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="num_animals",
                levels=list(range(1, 37)),
                description="Number of animals in question",
                lower_field_name="min_animals",
                upper_field_name="max_animals",
            ),
            RangeAttributeDefinition(
                name="num_instances",
                levels=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
                description="Number of instances of each animal",
                lower_field_name="min_instances",
                upper_field_name="max_instances",
                ensure_interval=True,
            ),
        )

register_dataset(DATASET_NAME, LegCountingDataset, LegCountingConfig, LegCountingCurriculum)
