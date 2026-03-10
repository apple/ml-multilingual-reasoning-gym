"""
Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""

"""Perform average / max pooling on a matrix"""

from dataclasses import dataclass
from random import Random
from typing import Any, Optional

import numpy as np

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import register_dataset
from ..multilingual.base_classes import MultilingualProceduralDataset
from ..config import DatasetConfig


DATASET_NAME = "pool_matrix"

@dataclass
class PoolMatrixConfig(DatasetConfig):
    """Configuration for Pool Matrix dataset generation"""

    min_rows: int = 2  # Minimum rows of the matrix
    max_rows: int = 10  # Maximum rows of the matrix
    min_cols: int = 2  # Minimum columns of the matrix
    max_cols: int = 10  # Maximum columns of the matrix
    min_pool_size: int = 1  # Minimum pooling size
    max_pool_size: int = 3  # Maximum pooling size
    languages: list[str] | str = "en"  # Add this field
    language_weights: Optional[list[float]] = None  # Add this optional field

    def validate(self):
        """Validate configuration parameters"""
        assert 2 <= self.min_rows, "min_rows must be at least 2"
        assert self.min_rows <= self.max_rows, "max_rows must be at least min_rows"
        assert 2 <= self.min_cols, "min_cols must be at least 2"
        assert self.min_cols <= self.max_cols, "max_cols must be at least min_cols"
        assert 1 <= self.min_pool_size, "min_pool_size must be at least 1"
        assert self.min_pool_size <= self.max_pool_size, "max_pool_size must be at least min_pool_size"

class PoolMatrixDataset(MultilingualProceduralDataset):
    """Generates Pool Matrix exercises with configurable difficulty"""

    def __init__(self, config: PoolMatrixConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _matrix_to_str(self, matrix: np.ndarray) -> str:
        """Get a string representation of the matrix"""
        return "\n".join(" ".join(str(round(x, 2)) for x in row) for row in matrix)

    def _max_pool(self, matrix: np.ndarray, pool_size: int) -> np.ndarray:
        """Perform max pooling on the matrix"""
        rows, cols = matrix.shape
        return np.array(
            [
                [np.max(matrix[i : i + pool_size, j : j + pool_size]) for j in range(0, cols, pool_size)]
                for i in range(0, rows, pool_size)
            ]
        )

    def _average_pool(self, matrix: np.ndarray, pool_size: int) -> np.ndarray:
        """Perform average pooling on the matrix"""
        rows, cols = matrix.shape
        return np.array(
            [
                [np.mean(matrix[i : i + pool_size, j : j + pool_size]) for j in range(0, cols, pool_size)]
                for i in range(0, rows, pool_size)
            ]
        )

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Score the answer based on the metadata"""

        if not isinstance(answer, str):
            return 0.0

        reward = 0.0
        try:
            oracle_answer = np.loadtxt(entry["answer"].splitlines(), dtype=np.float32)
            answer = np.loadtxt(answer.splitlines(), dtype=np.float32)
            if oracle_answer.shape == answer.shape and np.allclose(oracle_answer, answer, rtol=1e-2):
                reward = 1.0
            elif oracle_answer.shape == answer.shape:
                reward = 0.1
        except Exception:
            pass
        return reward

    def _generate_item(self, idx: int, language: str) -> dict[str, Any]:
        """Generate a single Pool Matrix question for a specific language."""
        rng = Random(self.seed + idx)
        np.random.seed(self.seed + idx)

        rows = rng.randint(self.config.min_rows, self.config.max_rows)
        cols = rng.randint(self.config.min_rows, self.config.max_cols)
        matrix = np.random.randint(0, 10, (rows, cols))
        matrix_str = self._matrix_to_str(matrix)

        pool_size = rng.randint(self.config.min_pool_size, self.config.max_pool_size)
        pool_type = rng.choice(["average", "max"])

        answer = self._average_pool(matrix, pool_size) if pool_type == "average" else self._max_pool(matrix, pool_size)
        answer_str = self._matrix_to_str(answer)

        # Get localized pool type name
        pool_type_localized = self._get_translation(f"pool_type_{pool_type}", language)

        # Get translated question template
        question = self._get_translation("question_template", language, 
                                       matrix=matrix_str, 
                                       pool_type=pool_type_localized, 
                                       pool_size=pool_size)

        return {
            "question": question,
            "answer": answer_str,
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "language": language,
                "matrix": matrix.tolist(),
                "pool_type": pool_type,
                "pool_size": pool_size,
                "solution": answer.tolist(),
                "rows": rows,
                "cols": cols,
                "difficulty": {
                    "rows": (self.config.min_rows, self.config.max_rows),
                    "cols": (self.config.min_cols, self.config.max_cols),
                    "pool_size": (self.config.min_pool_size, self.config.max_pool_size),
                },
            },
        }

class PoolMatrixCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(PoolMatrixCurriculum.__name__, PoolMatrixConfig)

        self._define_attributes(
            RangeAttributeDefinition(
                name="rows",
                levels=[10, 25, 50, 100],
                description="Board size",
                lower_field_name="min_rows",
                upper_field_name="max_rows",
            ),
            RangeAttributeDefinition(
                name="cols",
                levels=[10, 25, 50, 100],
                description="Board size",
                lower_field_name="min_cols",
                upper_field_name="max_cols",
            ),
            RangeAttributeDefinition(
                name="pool_size",
                levels=[3, 5, 7, 9],
                description="Pool size",
                lower_field_name="min_pool_size",
                upper_field_name="max_pool_size",
            ),
        )

register_dataset(DATASET_NAME, PoolMatrixDataset, PoolMatrixConfig, PoolMatrixCurriculum)
