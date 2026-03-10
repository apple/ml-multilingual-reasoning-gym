"""
Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""

import random
from dataclasses import dataclass
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..config import DatasetConfig
from ..factory import register_dataset
from ..multilingual.base_classes import MultilingualProceduralDataset

DATASET_NAME = "simple_geometry"


@dataclass
class SimpleGeometryConfig(DatasetConfig):
    """
    Configuration for generating basic geometry (angle-finding) tasks.
    Produces a random convex polygon with N sides, random angles
    for the first (N-1) sides, and asks the solver to find the last angle.
    """

    min_sides: int = 3  # Minimum number of sides (e.g. triangle)
    max_sides: int = 6  # Maximum number of sides (e.g. hexagon)
    min_angle: int = 10  # Minimum angle (in degrees) for each of the first (N-1) angles
    max_angle: int = 170  # Maximum angle (in degrees) for each of the first (N-1) angles
    languages: list[str] | str = "en"  # Supported languages
    language_weights: Optional[list[float]] = None  # Weights for language sampling

    def validate(self) -> None:
        """
        Validate configuration parameters.
        """
        assert self.min_sides >= 3, "min_sides must be at least 3 (triangle)."
        assert self.max_sides >= self.min_sides, "max_sides must be >= min_sides."
        assert 0 < self.min_angle < 180, "min_angle must be in (0, 180)."
        assert self.max_angle <= 179, "max_angle should be less than 180."
        assert self.max_angle >= self.min_angle, "max_angle must be >= min_angle."


class SimpleGeometryDataset(MultilingualProceduralDataset):
    """
    A dataset for simple polygon angle-finding tasks.
    We randomly choose the number of sides N within [min_sides, max_sides].
    We then generate (N-1) random angles (in degrees), ensuring their sum is
    strictly less than the total sum for an (N)-sided convex polygon (which is 180*(N-2)).
    The question asks for the missing angle; the answer is computed by subtracting the
    sum of known angles from 180*(N-2).
    """

    def __init__(self, config: SimpleGeometryConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _generate_item(self, idx: int, language: str) -> dict[str, Any]:
        """
        Generate a single geometry angle-finding item.

        Args:
            idx: The index of the item to generate
            language: The language for the generated question

        Returns:
            A dict with:
                - question: str
                - answer: str (the missing angle, as an integer or float in degrees)
                - metadata: dict (n_sides, angles, sum_of_known, missing_angle, etc.)
        """
        rng = random.Random(self.seed + idx)

        # Randomly pick the number of sides
        n_sides = rng.randint(self.config.min_sides, self.config.max_sides)

        # Total interior angle sum for a convex n_sides-gon
        total_sum = 180 * (n_sides - 2)

        # Generate (n_sides - 1) random angles, ensuring their sum < total_sum
        known_angles = self._generate_valid_angles(rng, n_sides, total_sum)

        # Missing angle
        missing_angle = total_sum - sum(known_angles)

        # Get localized angle separator
        angle_separator = self._get_translation("angle_separator", language)

        # Build the question string with localized separator
        angle_list_str = angle_separator.join(f"{a:.1f}°" for a in known_angles)

        # Choose a random prompt template and format with translations
        template_idx = rng.randint(1, 3)  # 3 templates available
        template_key = f"prompt_template_{template_idx}"

        prompt = self._get_translation(
            template_key, language, n_sides=n_sides, n_minus_1=n_sides - 1, angle_list=angle_list_str
        )

        # Round the missing angle to one decimal place or integer if it is very close to an integer
        # so that the answer remains consistent and clean
        missing_angle_rounded = round(missing_angle, 1)
        if abs(missing_angle_rounded - round(missing_angle_rounded)) < 1e-6:
            # If it is effectively an integer, keep it as int
            missing_angle_rounded = int(missing_angle_rounded)

        answer_str = str(missing_angle_rounded)

        return {
            "question": prompt,
            "answer": answer_str,
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "n_sides": n_sides,
                "known_angles": known_angles,
                "sum_of_known_angles": sum(known_angles),
                "missing_angle_raw": missing_angle,
                "missing_angle_rounded": missing_angle_rounded,
                "total_interior_sum": total_sum,
                "difficulty": {
                    "sides": (self.config.min_sides, self.config.max_sides),
                },
                "language": language,
            },
        }

    def _generate_valid_angles(self, rng: random.Random, n_sides: int, total_sum: int):
        """
        Generate (n_sides - 1) random angles in [min_angle, max_angle],
        ensuring the sum is strictly less than total_sum to keep a valid missing angle.
        We keep retrying until we find a valid set or reach a max attempt limit.
        """
        max_attempts = 100
        for _ in range(max_attempts):
            angles = []
            # We choose angles one by one
            for _ in range(n_sides - 1):
                angle = rng.randint(self.config.min_angle, self.config.max_angle)
                angles.append(float(angle))

            # Check if the sum is strictly less than total_sum
            if sum(angles) < total_sum:
                return angles

        # If we fail after max_attempts, raise an error
        raise ValueError(
            f"Could not generate valid angles for an {n_sides}-gon "
            f"with total sum {total_sum} within {max_attempts} attempts."
        )


class SimpleGeometryCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(SimpleGeometryCurriculum.__name__, SimpleGeometryConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="sides",
                levels=[5, 10, 15, 30],
                description="Number of sides in the polygon.",
                lower_field_name="min_sides",
                upper_field_name="max_sides",
                ensure_interval=True,
            )
        )


# Register the dataset so it can be accessed similarly to the others
register_dataset(DATASET_NAME, SimpleGeometryDataset, SimpleGeometryConfig, SimpleGeometryCurriculum)
