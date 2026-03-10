"""
Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""

from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import register_dataset
from ..multilingual.base_classes import MultilingualProceduralDataset
from ..config import DatasetConfig

DATASET_NAME = "arc_1d"

@dataclass
class Arc1DConfig(DatasetConfig):
    """Configuration for ARC 1D task generation"""

    min_size: int = 10  # Minimum grid size
    max_size: int = 30  # Maximum grid size
    num_train: int = 3  # Number of training examples
    languages: list[str] | str = "en"  # Add this field
    language_weights: Optional[list[float]] = None  # Add this optional field

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.min_size >= 8, "min_size must be >= 8"
        assert self.max_size >= self.min_size, "max_size must be >= min_size"
        assert self.num_train > 0, "num_train must be positive"
        assert self.size > 0, "size must be positive"

class Arc1DDataset(MultilingualProceduralDataset):
    """
    Generates ARC 1D tasks by randomly selecting from available task generators

    This dataset is a procedural variant of the 1D-ARC dataset which is described in the paper:
    `LLMs and the Abstraction and Reasoning Corpus:  Successes, Failures, and the Importance
    of Object-based Representations` (https://arxiv.org/abs/2305.18354)

    Ilya Sheprut (optozorax) created rust generators for most of the ARC 1d tasks. For
    reasoning-gym rust tasks were machine-converted to python via Sonnet.

    Ilya's original rust code can be found here: https://github.com/optozorax/arc_1d/
    """

    def __init__(self, config: Arc1DConfig):
        from .arc_1d_tasks import ARC_1D_TASKS

        super().__init__(config=config, seed=config.seed, size=config.size)
        self.ARC_1D_TASKS = ARC_1D_TASKS
        self.task_names = list(ARC_1D_TASKS.keys())

    def _generate_item(self, idx: int, language: str) -> dict[str, Any]:
        """Generate a single ARC 1D task with training examples for a specific language

        Args:
            idx: Index of the item to generate
            language: Language code for the task

        Returns:
            dict with keys:
                - question: str, the task description and examples
                - answer: str, the expected output format
                - metadata: dict with generation parameters
        """
        # Create deterministic RNG from base seed and idx
        rng = Random(self.seed + idx)

        # Select random task
        task_name = rng.choice(self.task_names)
        task_func, task_kwargs = self.ARC_1D_TASKS[task_name]

        # Generate training examples
        train_examples = []
        size = rng.randint(self.config.min_size, self.config.max_size)

        for _ in range(self.config.num_train):
            example = None
            while example is None:
                example = task_func(rng, size, **task_kwargs)

            train_examples.append(example)

        # Generate test example
        test_example = None
        while test_example is None:
            test_example = task_func(rng, size, **task_kwargs)

        # Format question
        # Build examples string
        examples_text = ""
        for i, example in enumerate(train_examples, 1):
            example_header = self._get_translation("example_header", language, number=i)
            input_label = self._get_translation("input_label", language)
            output_label = self._get_translation("output_label", language)
            
            examples_text += f"{example_header}\n"
            examples_text += input_label + " ".join(str(x) for x in example["input"]) + "\n"
            examples_text += output_label + " ".join(str(x) for x in example["output"]) + "\n\n"

        # Format test input
        test_input_text = " ".join(str(x) for x in test_example["input"])

        # Use single template with placeholders
        question = self._get_translation("question_template", language, 
                                       examples=examples_text.rstrip(),
                                       test_input=test_input_text)

        return {
            "question": question,
            "answer": " ".join(str(x) for x in test_example["output"]),
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "task_name": task_name,
                "size": size,
                "train_examples": train_examples,
                "test_example": test_example,
                "language": language,
                "difficulty": {
                    "size": (self.config.min_size, self.config.max_size),
                },
            },
        }

class Arc1DCurriculum(BaseCurriculum):
    """Curriculum for ARC 1D tasks"""

    def __init__(self):
        super().__init__(Arc1DCurriculum.__name__, Arc1DConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="size",
                levels=[10, 25, 50, 100],
                lower_field_name="min_size",
                upper_field_name="max_size",
                description="Grid size",
            )
        )

# Register the dataset
register_dataset(DATASET_NAME, Arc1DDataset, Arc1DConfig, Arc1DCurriculum)
