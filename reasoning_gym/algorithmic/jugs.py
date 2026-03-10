"""
Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""

import json
import math
from collections import deque
from dataclasses import dataclass
from functools import reduce
from random import Random
from typing import Any, Optional

from ..coaching import BaseCurriculum, ScalarAttributeDefinition
from ..factory import register_dataset
from ..multilingual.base_classes import MultilingualProceduralDataset
from ..config import DatasetConfig

DATASET_NAME = "jugs"

def min_moves_n(jug_capacities: list[int], target: int) -> Optional[int]:
    """
    Compute the minimum number of moves required to have exactly `target` gallons
    in any one jug for a puzzle with multiple jugs.
    The state is represented as a tuple (w1, w2, ..., wn), where each wi is the current
    amount in jug i.

    Allowed moves:
      - Fill jug i to its capacity.
      - Empty jug i.
      - Pour from jug i to jug j until jug i is empty or jug j is full.

    Returns the minimal move count if a solution exists, otherwise None.
    """
    n = len(jug_capacities)
    start = tuple([0] * n)
    queue = deque([(start, 0)])
    visited = set([start])

    while queue:
        state, moves = queue.popleft()

        # Check if any jug has the target amount.
        if any(w == target for w in state):
            return moves

        # Generate next states.
        next_states = []

        # 1. Fill any jug.
        for i in range(n):
            new_state = list(state)
            new_state[i] = jug_capacities[i]
            next_states.append(tuple(new_state))

        # 2. Empty any jug.
        for i in range(n):
            new_state = list(state)
            new_state[i] = 0
            next_states.append(tuple(new_state))

        # 3. Pour from one jug to another.
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if state[i] == 0 or state[j] == jug_capacities[j]:
                    continue
                new_state = list(state)
                # Maximum water that can be poured from i to j.
                amount = min(state[i], jug_capacities[j] - state[j])
                new_state[i] -= amount
                new_state[j] += amount
                next_states.append(tuple(new_state))

        # Add valid next states to the queue.
        for ns in next_states:
            if ns not in visited:
                visited.add(ns)
                queue.append((ns, moves + 1))

    return None

def generate_puzzle(rng: Random, num_jugs: int = 3, difficulty: int = 6, max_attempts: int = 10000) -> dict[str, Any]:
    """
    Generate a multi-jug water puzzle.

    Parameters:
      - num_jugs: number of jugs to use (>=2; default 3).
      - difficulty: minimal required moves for a solution.
      - max_attempts: maximum attempts to generate a puzzle meeting the difficulty.

    For a valid puzzle:
      - Each jug gets a random capacity (between 3 and 3+difficulty).
      - The target is chosen as one of the numbers 1 .. (max_capacity) that is a multiple
        of the gcd of all jug capacities.

    Returns a dictionary with:
       { "jug_capacities": [c1, c2, ...],
         "target": target,
         "min_moves": minimum moves required }.

    Raises a ValueError if no puzzle is generated after max_attempts.
    """
    for _ in range(max_attempts):
        # Generate capacities for each jug.
        jug_capacities = [rng.randint(3, 3 + difficulty) for _ in range(num_jugs)]
        max_cap = max(jug_capacities)
        # Compute gcd of all jug capacities.
        gcd_all = reduce(math.gcd, jug_capacities)
        # Possible targets are between 1 and max_cap that are multiples of gcd_all.
        possible_targets = [t for t in range(1, max_cap + 1) if t % gcd_all == 0]
        if not possible_targets:
            continue
        target = rng.choice(possible_targets)

        moves = min_moves_n(jug_capacities, target)
        if moves is not None and moves >= difficulty:
            return {"jug_capacities": jug_capacities, "target": target, "min_moves": moves}
    if moves is not None:
        # return valid jugs puzzle even if it doesn't necessarily conform to the curriculum
        print(
            f"Jugs: Could not generate a puzzle with difficulty at least {difficulty} using {num_jugs} jugs.", 
            {"jug_capacities": jug_capacities, "target": target, "min_moves": moves}
        )
        return {"jug_capacities": jug_capacities, "target": target, "min_moves": moves}
    raise ValueError(f"Could not generate a puzzle with difficulty at least {difficulty} using {num_jugs} jugs.")

def verify_solution(puzzle, moves, fill_action: str, empty_action: str, pour_action: str):
    """
    Verify a given solution for a multi-jug puzzle.

    The puzzle is a dictionary with keys:
      - "jug_capacities": list of capacities for each jug.
      - "target": the target amount that must be in any one jug.

    Moves should be a list of strings in the following formats:
      - "fill X": Fill jug X to its capacity.
      - "empty X": Empty jug X.
      - "pour X->Y": Pour water from jug X to jug Y.

    Jug labels are letters: jug 0 is "A", jug 1 is "B", etc.

    The function simulates the moves starting from all jugs empty.

    Returns a tuple (result, states) where:
      - result is True if, after executing all moves, at least one jug has exactly
        the target amount; otherwise False.
      - states is a list of state tuples after each move.
    """
    jug_capacities = puzzle["jug_capacities"]
    target = puzzle["target"]
    n = len(jug_capacities)

    # Map jug letters to indices (A->0, B->1, C->2, etc.)
    jug_map = {chr(ord("A") + i): i for i in range(n)}

    state = tuple([0] * n)
    states = [state]

    for move in moves:
        tokens = move.split()
        if tokens[0] == fill_action:
            # Move format: "fill_action X"
            jug = tokens[1]
            idx = jug_map[jug]
            state = list(state)
            state[idx] = jug_capacities[idx]
            state = tuple(state)
        elif tokens[0] == empty_action:
            # Move format: "empty_action X"
            jug = tokens[1]
            idx = jug_map[jug]
            state = list(state)
            state[idx] = 0
            state = tuple(state)
        elif tokens[0] == pour_action:
            # Move format: "pour_action X->Y"
            # Expect tokens[1] to be in the form "X->Y"
            parts = tokens[1].split("->")
            if len(parts) != 2:
                raise ValueError(f"Invalid pour move format: {move}")
            source, dest = parts
            i = jug_map[source]
            j = jug_map[dest]
            state = list(state)
            amount = min(state[i], jug_capacities[j] - state[j])
            state[i] -= amount
            state[j] += amount
            state = tuple(state)
        else:
            raise ValueError(f"Unknown move: {move}")
        states.append(state)

    return (any(w == target for w in state), states)

def generate_jug_solution(jug_capacities: tuple[int, int, int], target: int, 
                         fill_action: str, empty_action: str, pour_action: str) -> list[str]:
    """Solves the jug puzzle and returns a sequence of formatted steps."""
    capacities = list(jug_capacities)
    initial_state = (0, 0, 0)
    queue = deque([(initial_state, [])])
    visited = set()

    while queue:
        (state, path) = queue.popleft()

        if target in state:
            return path  # Solution found

        if state in visited:
            continue
        visited.add(state)

        for i in range(3):  # Iterate over each jug
            # Fill jug i
            new_state = list(state)
            new_state[i] = capacities[i]
            queue.append((tuple(new_state), path + [f"{fill_action} {chr(65 + i)}"]))

            # Empty jug i
            new_state = list(state)
            new_state[i] = 0
            queue.append((tuple(new_state), path + [f"{empty_action} {chr(65 + i)}"]))

            # Pour from jug i to jug j
            for j in range(3):
                if i != j:
                    new_state = list(state)
                    pour_amount = min(state[i], capacities[j] - state[j])
                    new_state[i] -= pour_amount
                    new_state[j] += pour_amount
                    queue.append((tuple(new_state), path + [f"{pour_action} {chr(65 + i)}->{chr(65 + j)}"]))

    return ["No solution"]  # No valid solution found

@dataclass
class JugsConfig(DatasetConfig):
    """Configuration for Jugs puzzle generation"""

    num_jugs: int = 3  # Number of jugs in the puzzle (affects puzzle complexity and solution space)
    difficulty: int = 10  # Minimum required moves to solve the puzzle. Also affects max jug capacity (3 + difficulty)
    languages: list[str] | str = "en"  # Add this field
    language_weights: Optional[list[float]] = None  # Add this optional field

    def validate(self):
        """Validate configuration parameters"""
        assert self.num_jugs > 2, "num_jugs must be gt 2"
        assert self.difficulty > 0, "difficulty must be gt 0"
        assert self.difficulty < 200, "difficulty must be lt 200"

class JugsDataset(MultilingualProceduralDataset):
    """Generates water jug puzzles inspired by [this scene from _Die Hard 3_](https://www.youtube.com/watch?v=6cAbgAaEOVE), with configurable parameters"""

    def __init__(self, config: JugsConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _generate_item(self, idx: int, language: str) -> dict[str, Any]:
        """Generate a single Jugs task for a specific language

        Args:
            idx: Index of the item to generate
            language: Language code (e.g., 'en')
            
        Returns:
            dict with keys:
                - question: str, the task description
                - answer: str, a solution string
                - metadata: dict with generation parameters
        """
        rng = Random(self.seed + idx)

        puzzle = generate_puzzle(rng, num_jugs=self.config.num_jugs, difficulty=self.config.difficulty)
        
        # Get localized action words
        fill_action = self._get_translation("fill_action", language)
        empty_action = self._get_translation("empty_action", language) 
        pour_action = self._get_translation("pour_action", language)
        
        solution = generate_jug_solution(puzzle["jug_capacities"], puzzle["target"], 
                                        fill_action, empty_action, pour_action)

        # Get localized jug formatting separators
        jug_separator = self._get_translation("jug_separator", language)
        capacity_separator = self._get_translation("capacity_separator", language)

        # Format jug capacities with localized separators
        jug_entries = []
        for i, cap in enumerate(puzzle["jug_capacities"]):
            jug_entries.append(f"{chr(ord('A')+i)}{capacity_separator}{cap}")
        cap_str = jug_separator.join(jug_entries)
        
        question = self._get_translation("puzzle_description", language, 
                                       cap_str=cap_str, 
                                       target=puzzle['target'],
                                       fill_action=fill_action,
                                       empty_action=empty_action,
                                       pour_action=pour_action)

        return {
            "question": question,
            "answer": json.dumps(solution, ensure_ascii=False),  # one possible solution
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "puzzle": puzzle,
                "difficulty": {
                    "num_jugs": self.config.num_jugs,
                    "difficulty": self.config.difficulty,
                },
                "language": language,
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Determine if the solution provided solves the Jugs task.

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
            danswer = json.loads(answer)
            
            # Get the language and localized action words for verification
            language = entry["metadata"]["language"]
            fill_action = self._get_translation("fill_action", language)
            empty_action = self._get_translation("empty_action", language)
            pour_action = self._get_translation("pour_action", language)
            
            valid, _ = verify_solution(entry["metadata"]["puzzle"], danswer, 
                                     fill_action, empty_action, pour_action)
            if not valid:
                return 0.01  # json parsable
            else:
                return 1.0  # Yay
        except Exception as e:
            return 0.0

class JugsCurriculum(BaseCurriculum):
    """Curriculum for Jugs puzzles"""

    def __init__(self):
        super().__init__(JugsCurriculum.__name__, JugsConfig)

        # Define attributes
        self._define_attributes(
            ScalarAttributeDefinition(
                name="num_jugs",
                field_name="num_jugs",
                levels=[3, 4, 5, 7],
                description="Number of jugs in the puzzle",
            ),
            ScalarAttributeDefinition(
                name="difficulty",
                field_name="difficulty",
                levels=[5, 10, 15, 20],
                description="Minimum required moves to solve the puzzle",
            ),
        )

register_dataset(DATASET_NAME, JugsDataset, JugsConfig, JugsCurriculum)
