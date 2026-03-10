"""
Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""

"""Find the shortest path between a start and end point in a grid"""

from collections import deque
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import register_dataset
from ..multilingual.base_classes import MultilingualProceduralDataset
from ..config import DatasetConfig

DATASET_NAME = "shortest_path"

@dataclass
class ShortestPathConfig(DatasetConfig):
    """Configuration for Shortest Path dataset generation"""

    min_rows: int = 5
    max_rows: int = 8
    min_cols: int = 5
    max_cols: int = 8
    p_blocked: float = 0.4
    languages: list[str] | str = "en"
    language_weights: Optional[list[float]] = None

    def validate(self):
        """Validate configuration parameters"""
        assert 1 <= self.min_rows, "min_rows must be at least 1"
        assert self.min_rows <= self.max_rows, "min_rows must be less than or equal to max_rows"
        assert 1 <= self.min_cols, "min_cols must be at least 1"
        assert self.min_cols <= self.max_cols, "min_cols must be less than or equal to max_cols"
        assert 0 <= self.p_blocked <= 1, "p_blocked must be between 0 and 1"

class ShortestPathDataset(MultilingualProceduralDataset):
    """Generates Shortest Path exercises with configurable difficulty"""

    def __init__(self, config: ShortestPathConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _get_grid(self, rng: Random, rows: int, cols: int) -> list[list[str]]:
        """Generate a random grid with open and blocked cells"""
        grid = [["X" if rng.random() < self.config.p_blocked else "O" for _ in range(cols)] for _ in range(rows)]

        start_r, start_c = rng.randint(0, rows - 1), rng.randint(0, cols - 1)
        grid[start_r][start_c] = "*"

        while True:
            end_r, end_c = rng.randint(0, rows - 1), rng.randint(0, cols - 1)
            if (end_r, end_c) != (start_r, start_c):
                grid[end_r][end_c] = "#"
                break

        return grid

    def _matrix_to_str(self, matrix: list[list[int]]) -> str:
        """Get a string representation of the matrix"""
        return "\n".join(" ".join(str(x) for x in row) for row in matrix)

    def _get_answer(self, matrix: list[list[str]]) -> list[str]:
        """Run BFS to find the shortest path"""
        ROWS, COLS = len(matrix), len(matrix[0])
        DIRS = [(0, 1, "→"), (1, 0, "↓"), (0, -1, "←"), (-1, 0, "↑")]

        start_r, start_c = next((r, c) for r in range(ROWS) for c in range(COLS) if matrix[r][c] == "*")
        queue = deque([(start_r, start_c, [])])
        visited = set((start_r, start_c))

        while queue:
            r, c, path = queue.popleft()
            for dr, dc, direction in DIRS:
                new_r, new_c = r + dr, c + dc
                if 0 <= new_r < ROWS and 0 <= new_c < COLS and (new_r, new_c) not in visited:
                    new_path = path + [direction]
                    if matrix[new_r][new_c] == "#":
                        return new_path
                    if matrix[new_r][new_c] == "O":
                        visited.add((new_r, new_c))
                        queue.append((new_r, new_c, new_path))

        return []

    def _is_valid_path(self, matrix: list[list[str]], path: list[str]) -> bool:
        """Verifies the path goes from * to # without crossing X cells"""
        ROWS, COLS = len(matrix), len(matrix[0])
        DIRS = {"→": (0, 1), "↓": (1, 0), "←": (0, -1), "↑": (-1, 0)}

        start_r, start_c = next((r, c) for r in range(ROWS) for c in range(COLS) if matrix[r][c] == "*")
        end_r, end_c = next((r, c) for r in range(ROWS) for c in range(COLS) if matrix[r][c] == "#")

        r, c = start_r, start_c
        for direction in path:
            if direction not in DIRS:
                return False  # Invalid direction
            dr, dc = DIRS[direction]
            r, c = r + dr, c + dc
            if not (0 <= r < ROWS and 0 <= c < COLS):
                return False
            if matrix[r][c] == "X":
                return False

        return (r, c) == (end_r, end_c)

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Overwrite this method in derived classes if a single oracle answer is not available."""
        if isinstance(answer, str) and len(answer) > 0:
            oracle_answer = entry["answer"].strip()
            answer = answer.strip()

            # Exact answer
            if answer == oracle_answer:
                return 1.0

            matrix = entry["metadata"]["matrix"]
            answer = answer.split()
            oracle_answer = oracle_answer.split()

            # Path is valid and has the same length as the oracle answer
            if self._is_valid_path(matrix, answer) and len(answer) == len(oracle_answer):
                return 1.0

            # Path is valid but has a larger length than the oracle answer
            elif self._is_valid_path(matrix, answer):
                return 0.5

        return 0.0

    def _generate_item(self, idx: int, language: str) -> dict[str, Any]:
        """Generate a single Shortest Path question in the specified language"""
        rng = Random(self.seed + idx)

        rows = rng.randint(self.config.min_rows, self.config.max_rows)
        cols = rng.randint(self.config.min_cols, self.config.max_cols)
        matrix = self._get_grid(rng, rows, cols)
        matrix_str = self._matrix_to_str(matrix)
        answer = self._get_answer(matrix)
        answer_str = " ".join(answer) if answer else self._get_translation("infeasible", language)

        question = self._get_translation("question_template", language, 
                                        grid=matrix_str, 
                                        infeasible=self._get_translation("infeasible", language))

        return {
            "question": question,
            "answer": answer_str,
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "matrix": matrix,
                "solution": answer,
                "difficulty": {
                    "rows": (self.config.min_rows, self.config.max_rows),
                    "cols": (self.config.min_cols, self.config.max_cols),
                },
                "language": language,
            },
        }

class ShortestPathCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(ShortestPathCurriculum.__name__, ShortestPathConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="rows",
                levels=[10, 25, 50, 100],
                description="Number of rows in the grid",
                lower_field_name="min_rows",
                upper_field_name="max_rows",
            ),
            RangeAttributeDefinition(
                name="cols",
                levels=[10, 25, 50, 100],
                description="Number of columns in the grid",
                lower_field_name="min_cols",
                upper_field_name="max_cols",
            ),
        )

register_dataset(DATASET_NAME, ShortestPathDataset, ShortestPathConfig, ShortestPathCurriculum)
