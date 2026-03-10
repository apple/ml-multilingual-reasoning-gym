"""
Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""

from dataclasses import dataclass, field
from random import Random
from string import Template
from typing import Any, Optional

from ..coaching import BaseCurriculum, ScalarAttributeDefinition
from ..factory import register_dataset
from ..multilingual.base_classes import MultilingualProceduralDataset
from ..utils import StrEnum
from ..config import DatasetConfig

DATASET_NAME = "aiw"

class TaskType(StrEnum):
    """Defines the type of task for the Alice in Wonderland dataset."""

    SIBLINGS = "siblings"
    FRIENDS = "friends"
    COLLEAGUES = "colleagues"  # Added colleagues task

@dataclass
class AliceInWonderlandConfig(DatasetConfig):
    """Configuration options for the Alice in Wonderland dataset.

    Attributes:
        male_names (list[str]): List of male names to use in questions.
        female_names (list[str]): List of female names to use in questions. Must include 'Alice'.
        task_types (list[TaskType]): List of task types to include in dataset.
        seed (Optional[int]): Seed for random number generation.
        size (int): Number of samples in the dataset.
        max_entities (int): Max number of siblings/friends/colleagues in questions.
        languages (list[str] | str): Languages to support in the dataset.
        language_weights (Optional[list[float]]): Weights for language selection.
    """

    male_names: list[str] = field(
        default_factory=lambda: [
            "James",
            "John",
            "Robert",
            "Michael",
            "William",
            "David",
            "Richard",
            "Joseph",
            "Thomas",
            "Charles",
            "Bob",
        ]
    )
    female_names: list[str] = field(
        default_factory=lambda: [
            "Mary",
            "Patricia",
            "Jennifer",
            "Linda",
            "Elizabeth",
            "Barbara",
            "Susan",
            "Jessica",
            "Sarah",
            "Margaret",
            "Alice",
        ]
    )
    task_types: list[TaskType] = field(
        default_factory=lambda: [TaskType.SIBLINGS, TaskType.FRIENDS, TaskType.COLLEAGUES]  # Added Colleagues
    )
    task_type_weights: list[float] = field(default_factory=lambda: [1 / 3, 1 / 3, 1 / 3])
    max_entities: int = 6  # Added max_entities
    languages: list[str] | str = "en"
    language_weights: Optional[list[float]] = None

    def validate(self) -> None:
        """Validates the configuration parameters."""
        assert len(self.male_names) > 0, "must provide male names"
        assert len(self.female_names) > 0, "must provide female names"
        assert "Alice" in self.female_names, "'Alice' must be in female names"
        assert len(self.task_types) > 0, "must provide at least one task type"
        assert self.max_entities > 0, "max_entities must be positive"

class AliceInWonderlandDataset(MultilingualProceduralDataset):
    """
    A procedural dataset inspired by the "Alice in Wonderland" paper.

    The dataset is inspired by the following paper:
       @inproceedings{nezhurina2024alice,
       title={Alice in Wonderland: Simple Tasks Reveal Severe Generalization and
              Basic Reasoning Deficits in State-Of-the-Art Large Language Models},
       author={Marianna Nezhurina and Lucia Cipolina-Kun and Mehdi Cherti and
              Jenia Jitsev},
       booktitle={NeurIPS 2024 Workshop on Scientific Methods for Understanding
                  Deep Learning},
       year={2024},
       url={https://openreview.net/forum?id=Mkl7dzjYiW}
       }

    """

    def __init__(self, config: AliceInWonderlandConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _generate_item(self, idx: int, language: str) -> dict[str, Any]:
        """Generates a single Alice in Wonderland question.

        Args:
            idx (int): Index of the item to generate.
            language (str): Language code for the generated item.

        Returns:
            dict[str, Any]: A dictionary containing the generated question, the right answer
                and metadata including language.
        """
        rng = Random(self.seed + idx)
        task_type = rng.choices(self.config.task_types, weights=self.config.task_type_weights, k=1)[0]
        female_name = rng.choice(self.config.female_names)
        male_name = rng.choice(self.config.male_names)

        if task_type == TaskType.SIBLINGS:
            num_brothers = rng.randint(1, self.config.max_entities)
            num_sisters = rng.randint(1, self.config.max_entities)

            answer = num_sisters + 1
            template_key = f"siblings_template_{rng.randint(1, 2)}"
            question = self._get_translation(
                template_key, 
                language,
                female_name=female_name,
                male_name=male_name,
                num_brothers=num_brothers,
                num_sisters=num_sisters,
            )
        elif task_type == TaskType.FRIENDS:
            num_male = rng.randint(1, self.config.max_entities)
            num_female = rng.randint(1, self.config.max_entities)

            answer = num_female + 1
            question = self._get_translation(
                "friends_template",
                language,
                female_name=female_name,
                male_name=male_name,
                num_male=num_male,
                num_female=num_female,
            )
        elif task_type == TaskType.COLLEAGUES:
            num_male_colleagues_alice_circle = rng.randint(1, self.config.max_entities)
            num_female_colleagues_alice_circle = rng.randint(1, self.config.max_entities)
            num_male_colleagues_bob_circle = rng.randint(1, self.config.max_entities)
            num_female_colleagues_bob_circle = rng.randint(1, self.config.max_entities)

            answer = num_female_colleagues_alice_circle + 1
            question = self._get_translation(
                "colleagues_template",
                language,
                female_name=female_name,
                male_name=male_name,
                num_male_colleagues_alice_circle=num_male_colleagues_alice_circle,
                num_female_colleagues_alice_circle=num_female_colleagues_alice_circle,
                num_male_colleagues_bob_circle=num_male_colleagues_bob_circle,
                num_female_colleagues_bob_circle=num_female_colleagues_bob_circle,
            )

        return {
            "question": question,
            "answer": str(answer),
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "task_type": task_type.value,
                "language": language,
                "difficulty": {
                    "task_type_weights": self.config.task_type_weights,
                    "num_entities": self.config.max_entities,
                },
            },
        }

class AliceInWonderlandCurriculum(BaseCurriculum):
    """Curriculum for the Alice in Wonderland dataset."""

    def __init__(self):
        super().__init__(AliceInWonderlandCurriculum.__name__, AliceInWonderlandConfig)
        self._define_attributes(
            ScalarAttributeDefinition(
                name="task_type_weights",
                field_name="task_type_weights",
                description="The weight of the task type",
                levels=[
                    [1.0, 0.0, 0.0],
                    [0.9, 0.05, 0.05],
                    [0.7, 0.15, 0.15],
                    [0.6, 0.2, 0.2],
                    [0.5, 0.25, 0.25],
                    [0.4, 0.3, 0.3],
                    [0.3, 0.35, 0.35],
                    [0.2, 0.4, 0.4],
                    [0.1, 0.45, 0.45],
                ],
            ),
            ScalarAttributeDefinition(
                name="num_entities",
                field_name="max_entities",
                description="The number of entities in the question",
                levels=list(range(4, 18, 2)),
            ),
        )

register_dataset(DATASET_NAME, AliceInWonderlandDataset, AliceInWonderlandConfig, AliceInWonderlandCurriculum)
