"""
Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""

import random
from dataclasses import dataclass
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import register_dataset
from ..multilingual.base_classes import MultilingualProceduralDataset
from ..utils import StrEnum
from ..config import DatasetConfig

class Color(StrEnum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    YELLOW = "yellow"
    WHITE = "white"
    ORANGE = "orange"
    PURPLE = "purple"
    PINK = "pink"
    BROWN = "brown"
    GRAY = "gray"
    CYAN = "cyan"
    MAGENTA = "magenta"
    GOLD = "gold"
    SILVER = "silver"
    INDIGO = "indigo"
    VIOLET = "violet"

class Side(StrEnum):
    TOP = "top"
    RIGHT = "right"
    FRONT = "front"
    LEFT = "left"
    BACK = "back"
    BOTTOM = "bottom"

DATASET_NAME = "color_cube_rotation"

@dataclass
class Cube:
    """Represents a cube with colored sides"""

    colors: dict[Side, Color]

    def rotate_front_to_top(self) -> None:
        """Rotate cube so front face becomes top"""
        old = self.colors.copy()
        self.colors[Side.TOP] = old[Side.FRONT]
        self.colors[Side.FRONT] = old[Side.BOTTOM]
        self.colors[Side.BOTTOM] = old[Side.BACK]
        self.colors[Side.BACK] = old[Side.TOP]
        # Right and left stay in place

    def rotate_right_to_top(self) -> None:
        """Rotate cube so right face becomes top"""
        old = self.colors.copy()
        self.colors[Side.TOP] = old[Side.RIGHT]
        self.colors[Side.RIGHT] = old[Side.BOTTOM]
        self.colors[Side.BOTTOM] = old[Side.LEFT]
        self.colors[Side.LEFT] = old[Side.TOP]
        # Front and back stay in place

    def rotate_back_to_top(self) -> None:
        """Rotate cube so back face becomes top"""
        old = self.colors.copy()
        self.colors[Side.TOP] = old[Side.BACK]
        self.colors[Side.BACK] = old[Side.BOTTOM]
        self.colors[Side.BOTTOM] = old[Side.FRONT]
        self.colors[Side.FRONT] = old[Side.TOP]
        # Right and left stay in place

    def rotate_left_to_top(self) -> None:
        """Rotate cube so left face becomes top"""
        old = self.colors.copy()
        self.colors[Side.TOP] = old[Side.LEFT]
        self.colors[Side.LEFT] = old[Side.BOTTOM]
        self.colors[Side.BOTTOM] = old[Side.RIGHT]
        self.colors[Side.RIGHT] = old[Side.TOP]
        # Front and back stay in place

    def rotate_bottom_to_top(self) -> None:
        """Rotate cube so bottom face becomes top"""
        old = self.colors.copy()
        self.colors[Side.TOP] = old[Side.BOTTOM]
        self.colors[Side.BOTTOM] = old[Side.TOP]
        self.colors[Side.FRONT] = old[Side.BACK]
        self.colors[Side.BACK] = old[Side.FRONT]
        # Right and left stay in place

@dataclass
class ColorCubeRotationConfig(DatasetConfig):
    """Configuration for color cube rotation task generation"""

    min_rotations: int = 1
    max_rotations: int = 3
    languages: list[str] | str = "en"
    language_weights: Optional[list[float]] = None

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.min_rotations > 0, "min_rotations must be positive"
        assert self.max_rotations >= self.min_rotations, "max_rotations must be >= min_rotations"

class ColorCubeRotationDataset(MultilingualProceduralDataset):
    """Generates color cube rotation reasoning tasks"""

    def __init__(self, config: ColorCubeRotationConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _generate_item(self, idx: int, language: str) -> dict[str, Any]:
        """Generate a single color cube rotation task for the given language.
        
        Args:
            idx: Index of the item to generate
            language: Language code for the task
            
        Returns:
            Dictionary containing question, answer, and metadata
        """
        rng = random.Random(self.seed + idx)

        # Generate initial cube state
        cube = self._generate_cube(rng)
        initial_state = cube.colors.copy()

        # Generate sequence of rotations
        num_rotations = rng.randint(self.config.min_rotations, self.config.max_rotations)
        rotations = []

        # Keep trying until we have at least one valid rotation
        while len(rotations) < num_rotations:
            # Get all sides except TOP
            available_sides = [s for s in Side if s != Side.TOP]
            from_side = rng.choice(available_sides)
            rotations.append(from_side)
            self._rotate_to_top(cube, from_side)

        # Select target side for question
        target_side = rng.choice(list(Side))

        # Generate story
        story = self._generate_story(initial_state, rotations, target_side, language, rng)
        
        # Get translated answer color
        answer_color = self._get_translation(f"color_{cube.colors[target_side].value}", language)

        return {
            "question": story,
            "answer": answer_color,
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "language": language,
                "initial_state": {k.value: v.value for k, v in initial_state.items()},
                "rotations": [r.value for r in rotations],
                "target_side": target_side.value,
                "num_rotations": num_rotations,
                "difficulty": {
                    "rotations": (self.config.min_rotations, self.config.max_rotations),
                },
            },
        }

    def _generate_cube(self, rng: random.Random) -> Cube:
        """Generate a cube with random colors"""
        colors = list(Color)
        rng.shuffle(colors)  # Randomize color order
        return Cube(dict(zip(Side, colors)))

    def _rotate_to_top(self, cube: Cube, from_side: Side) -> None:
        """Rotate cube so that given side becomes top"""
        rotation_map = {
            Side.FRONT: cube.rotate_front_to_top,
            Side.RIGHT: cube.rotate_right_to_top,
            Side.BACK: cube.rotate_back_to_top,
            Side.LEFT: cube.rotate_left_to_top,
            Side.BOTTOM: cube.rotate_bottom_to_top,
        }
        if from_side in rotation_map:
            rotation_map[from_side]()

    def _generate_story(
        self,
        initial_state: dict[Side, Color],
        rotations: list[Side],
        target_side: Side,
        language: str,
        rng: random.Random,
    ) -> str:
        """Generate story describing cube state and rotations"""
        # Describe initial state
        story_parts = [self._get_translation("initial_state_intro", language)]
        for side in Side:
            translated_color = self._get_translation(f"color_{initial_state[side].value}", language)
            translated_side = self._get_translation(f"side_{side.value}", language)
            side_description = self._get_translation(
                "side_description", language, color=translated_color, side=translated_side
            )
            story_parts.append(side_description)

        # Describe rotations
        rotation_templates = [
            "rotation_first",
            "rotation_then",
            "rotation_after_that", 
            "rotation_now",
            "rotation_next"
        ]

        for i, from_side in enumerate(rotations):
            template_key = rotation_templates[0] if i == 0 else rng.choice(rotation_templates[1:])
            translated_from_side = self._get_translation(f"side_{from_side.value}", language)
            rotation_text = self._get_translation(template_key, language, side=translated_from_side)
            story_parts.append(f"\n{rotation_text}")

        # Ask question
        translated_target_side = self._get_translation(f"side_{target_side.value}", language)
        final_question = self._get_translation(
            "final_question", language, target_side=translated_target_side
        )
        story_parts.append(f"\n{final_question}")

        return "\n".join(story_parts)

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        reward = 0.0
        if answer is not None:
            try:
                answer_formatted = answer.strip().lower()
                solved = answer_formatted == entry["answer"].strip().lower()
                if solved:
                    reward = 1.0
                else:
                    reward = 0.01
            except Exception:
                reward = 0.01
        return reward

class ColorCubeRotationCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(
            ColorCubeRotationCurriculum.__name__, ColorCubeRotationConfig
        )

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="rotations",
                levels=[1, 5, 10, 50, 100],
                description="Number of rotations to perform on the cube",
                lower_field_name="min_rotations",
                upper_field_name="max_rotations",
                ensure_interval=True,
            )
        )

register_dataset(DATASET_NAME, ColorCubeRotationDataset, ColorCubeRotationConfig, ColorCubeRotationCurriculum)
