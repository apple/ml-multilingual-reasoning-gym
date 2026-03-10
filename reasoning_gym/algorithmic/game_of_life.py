"""
Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""

import json
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

import cellpylib as cpl

from ..coaching import BaseCurriculum, ScalarAttributeDefinition
from ..factory import register_dataset
from ..config import DatasetConfig
from ..multilingual.base_classes import MultilingualProceduralDataset

DATASET_NAME = "game_of_life"

@dataclass
class GameOfLifeConfig(DatasetConfig):
    """Configuration for Game of Life puzzle generation"""

    grid_size_x: int = 10
    grid_size_y: int = 10
    filled_cells_weights: float = 0.1
    filled_cells: Optional[int] = None
    simulation_steps: int = 1
    languages: list[str] | str = "en"
    language_weights: Optional[list[float]] = None

    def __post_init__(self):
        # Calculate filled_cells if not explicitly provided
        if self.filled_cells is None:
            self.filled_cells = int(self.filled_cells_weights * self.grid_size_x * self.grid_size_y)

    def validate(self):
        """Validate configuration parameters"""
        assert 3 <= self.grid_size_x <= 999, "grid_size_x must be between 0 and 999"
        assert 3 <= self.grid_size_y <= 999, "grid_size_y must be between 0 and 999"
        assert self.simulation_steps >= 0, "simulation_steps must be gte 0"
        assert 0.0 <= self.filled_cells_weights <= 1.0, "filled_cells_weights must be between 0.0 and 1.0"
        assert self.filled_cells <= self.grid_size_x * self.grid_size_y, "filled_cells must fit in x times y"

class GameOfLifeDataset(MultilingualProceduralDataset):
    """Generates Game of Life games with configurable parameters"""

    def __init__(self, config: GameOfLifeConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _generate_item(self, idx: int, language: str) -> dict[str, Any]:
        """Generate a single GameOfLife task for a specific language

        Args:
            idx: The index of the item to generate
            language: The language code for the generated item

        Returns:
            dict with keys:
                - question: str, the task description
                - answer: str, a solution string
                - metadata: dict with generation parameters
        """
        rng = Random(self.seed + idx)

        # Make the board
        board = cpl.init_simple2d(self.config.grid_size_x, self.config.grid_size_y)
        board[:, :, :] = 0

        # Add the cells
        for i in range(0, self.config.filled_cells):
            rx = rng.randint(0, self.config.grid_size_x - 1)
            ry = rng.randint(0, self.config.grid_size_y - 1)
            board[:, rx, ry] = 1

        # Simulate the result to get the answer
        evolved = cpl.evolve2d(
            board,
            timesteps=self.config.simulation_steps + 1,
            apply_rule=cpl.game_of_life_rule,
            memoize="recursive",
        )

        rows = [json.dumps(board[0, i].tolist(), separators=(",", ":")) for i in range(board.shape[1])]
        board_str = "[" + ",\n ".join(rows) + "]"

        final_step = evolved[-1]
        final_step_list = final_step.tolist()
        result_str = json.dumps(final_step_list, separators=(",", ":"))

        return {
            "question": self._get_translation(
                "question_template", language, 
                simulation_steps=self.config.simulation_steps, 
                board=board_str
            ),
            "answer": result_str,
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "grid_size_x": self.config.grid_size_x,
                "grid_size_y": self.config.grid_size_y,
                "filled_cells": self.config.filled_cells,
                "simulation_steps": self.config.simulation_steps,
                "language": language,
                "difficulty": {
                    "grid_size_x": self.config.grid_size_x,
                    "grid_size_y": self.config.grid_size_y,
                    "filled_cells_weights": self.config.filled_cells_weights,
                    "simulation_steps": self.config.simulation_steps,
                },
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Determine if the solution provided solves the GoL task.

        The function awards 1.0 for a correct answer.

        Args:
            answer (Optional[str]): The user's answer.
            entry (dict[str, Any]): The original dataset entry containing the correct answer.

        Returns:
            float: The computed score between 0.0 and 1.0.
        """

        if answer is None:
            return 0.0

        try:
            ans_arr = json.loads(answer)
            correct_arr = json.loads(entry["answer"])
        except Exception:
            return 0.0

        total_cells = 0
        correct_cells = 0

        # Determine if the array is 2D (i.e. a list of lists)
        is_2d = correct_arr and isinstance(correct_arr[0], list)

        if is_2d:
            # Iterate over rows and columns of the expected grid.
            for i, expected_row in enumerate(correct_arr):
                for j, expected_value in enumerate(expected_row):
                    total_cells += 1
                    try:
                        if ans_arr[i][j] == expected_value:
                            correct_cells += 1
                    except (IndexError, TypeError):
                        # Either the row or the cell is missing, treat as incorrect.
                        pass
        else:
            # 1D array case.
            for i, expected_value in enumerate(correct_arr):
                total_cells += 1
                try:
                    if ans_arr[i] == expected_value:
                        correct_cells += 1
                except IndexError:
                    pass

        # If for some reason there are no cells, return 0.0.
        if total_cells == 0:
            return 0.0

        # Each cell contributes equally.
        return correct_cells / total_cells

class GameOfLifeCurriculum(BaseCurriculum):
    """Curriculum for Game of Life dataset"""

    def __init__(self):
        super().__init__(GameOfLifeCurriculum.__name__, GameOfLifeConfig)

        # Define attributes
        self._define_attributes(
            ScalarAttributeDefinition(
                name="grid_size_x",
                field_name="grid_size_x",
                levels=[10, 25, 50, 100],
                description="Grid size in the x direction",
            ),
            ScalarAttributeDefinition(
                name="grid_size_y",
                field_name="grid_size_y",
                levels=[10, 25, 50, 100],
                description="Grid size in the y direction",
            ),
            # Filled cells should be 10%, 20%, 30%, 50% of the grid_size_x * grid_size_y
            ScalarAttributeDefinition(
                name="filled_cells_weights",
                field_name="filled_cells_weights",
                levels=[0.1, 0.2, 0.5, 0.8],
                description="Percentage of filled cells in the grid",
            ),
            ScalarAttributeDefinition(
                name="simulation_steps",
                field_name="simulation_steps",
                levels=[1, 2, 5, 10],
                description="Number of simulation steps to run",
            ),
        )

register_dataset(DATASET_NAME, GameOfLifeDataset, GameOfLifeConfig, GameOfLifeCurriculum)
