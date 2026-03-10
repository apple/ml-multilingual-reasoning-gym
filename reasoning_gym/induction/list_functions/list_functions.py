"""
Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""

"""List functions generators"""

from dataclasses import dataclass
from random import Random
from typing import Any, Callable, Optional

from reasoning_gym.coaching.attributes import ScalarAttributeDefinition
from reasoning_gym.coaching.base_curriculum import BaseCurriculum
from reasoning_gym.factory import register_dataset
from reasoning_gym.multilingual.base_classes import MultilingualProceduralDataset
from reasoning_gym.config import DatasetConfig

DATASET_NAME = "list_functions"

@dataclass
class ListFunctionsDatasetConfig(DatasetConfig):
    """Configuration for List function generators."""
    mock_for_curriculum: int = 1
    languages: list[str] | str = "en"
    language_weights: Optional[list[float]] = None

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.size > 0, "size must be positive"

tasks = list(range(17))

class ListFunctionsDataset(MultilingualProceduralDataset):

    def __init__(self, config: ListFunctionsDatasetConfig):
        super().__init__(config, config.seed, config.size)
        self._generators: dict[int, Callable[[Random, float], dict[str, Any]]] = None  # initially None, lazy loading
        # self.task_indices = Random(self.seed).choices(tasks, k=self.size)

    @property
    def generators(self) -> dict[int, Callable[[Random, float], dict[str, Any]]]:
        """Lazy load generators only when first accessed"""
        if self._generators is None:
            self._generators = self._load_generators()
        return self._generators

    def _load_generators(self):
        """
        Generates mapper from task identifiers (keys) to example generator functions
        """
        from . import generators

        def strip_prefix(s: str, prefix: str) -> str:
            return s[len(prefix) :]

        prefix = "generate_"
        gs = {}
        for n in dir(generators):
            if n.startswith(prefix):
                gs[int(strip_prefix(n, prefix))] = getattr(generators, n)
        return gs

    def _generate_item(self, idx: int, language: str) -> dict[str, Any]:
        """Generate a single induction-based list function dataset item in the specified language."""
        rng = Random(self.seed + idx)
        generator_idx = rng.choice(tasks)
        # generator_idx = self.task_indices[idx]
        generator = self.generators[generator_idx]
        examples = generator(rng)
        entry = examples.popitem()
        input = entry[0]
        output = entry[1]
        formatted_examples = ""
        colon = self._get_translation("colon", language)
        for index, key in enumerate(examples):
            input_label = self._get_translation("input_label", language, number=index + 1)
            output_label = self._get_translation("output_label", language, number=index + 1)
            formatted_examples += f"""{input_label}{colon}{key}
{output_label}{colon}{examples[key]}
"""
        
        question = self._get_translation("prompt_template", language, 
                                       examples=formatted_examples, input=input)
        
        return {
            "question": question,
            "answer": output,
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "language": language,
                "difficulty": {
                    "mock": 1
                }
            },
        }


class ListFunctionsCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(ListFunctionsCurriculum.__name__, ListFunctionsDatasetConfig)
        self._define_attributes(
            ScalarAttributeDefinition(
                name="mock_for_curriculum",
                field_name="mock_for_curriculum",
                levels=[1, 1, 1, 1],
                description="mock_for_curriculum",
            ),
        )


register_dataset(DATASET_NAME, ListFunctionsDataset, ListFunctionsDatasetConfig, ListFunctionsCurriculum)
