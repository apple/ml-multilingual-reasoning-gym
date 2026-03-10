"""
Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""

"""Base classes for multilingual dataset support."""

from abc import abstractmethod
from random import Random
from typing import Any, Optional

from ..dataset import ProceduralDataset
from ..config import DatasetConfig
from .translation_manager import TranslationManager


class MultilingualProceduralDataset(ProceduralDataset):
    """Base class for datasets that support multiple languages.

    This class extends ProceduralDataset to add multilingual support through:
    - Language selection logic (deterministic per sample)
    - Translation management and validation
    - Helper methods for accessing translations
    """

    def __init__(
        self, config: DatasetConfig, seed: Optional[int] = None, size: int = 500
    ):
        """Initialize multilingual dataset.

        Args:
            config: Dataset configuration (must include languages field)
            seed: Random seed for reproducible generation
            size: Number of examples to generate
        """
        super().__init__(
            config=config,
            seed=seed or getattr(config, "seed", None),
            size=size or getattr(config, "size", 500),
        )

        # Normalize languages to list
        if isinstance(config.languages, str):
            self.languages = [config.languages]
        else:
            self.languages = list(config.languages)

        # Initialize translation manager
        self._translation_manager = TranslationManager()

        # Validate that translations exist for all requested languages
        self._validate_translations()

    def _validate_translations(self):
        """Validate that translations exist for all requested languages.

        Raises:
            FileNotFoundError: If translations are missing for any requested language
        """
        available_languages = self._translation_manager.get_available_languages(
            self.task_name
        )

        missing_languages = [
            lang for lang in self.languages if lang not in available_languages
        ]
        if missing_languages:
            raise FileNotFoundError(
                f"Translations for task '{self.task_name}' not available in languages: {missing_languages}. "
                f"Available languages: {available_languages}"
            )

    def _get_sample_language(self, idx: int) -> str:
        """Deterministically select language for a specific sample.

        Args:
            idx: Sample index

        Returns:
            Selected language code for this sample
        """
        if len(self.languages) == 1:
            return self.languages[0]

        # Use sample index + base seed for deterministic selection
        rng = Random(self.seed + idx)
        weights = self.config.language_weights or [1.0] * len(self.languages)
        return rng.choices(self.languages, weights=weights)[0]

    @abstractmethod
    def _generate_item(self, idx: int, language: str) -> dict[str, Any]:
        """Generate a dataset item for the given index and language.

        Args:
            idx: Index of the item to generate
            language: Language code for this sample

        Returns:
            Dictionary containing at least:
                - question: str - The question text
                - answer: str - The correct answer
                - metadata: dict - Should include "language": language
        """
        raise NotImplementedError("Subclasses must implement _generate_item")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Generate item with automatic language selection.

        Args:
            idx: Index of the item to generate

        Returns:
            Dictionary containing question, answer, and metadata with language info
        """
        language = self._get_sample_language(idx)
        return self._generate_item(idx, language)

    def _get_translation(self, key: str, language: str, **kwargs) -> str:
        """Get translation for specified key and language.

        Args:
            key: Translation key (e.g., 'question_template', 'hint_template')
            language: Language code
            **kwargs: Parameters to pass to the translation function/template

        Returns:
            Translated text

        Raises:
            FileNotFoundError: If translation is not available
            KeyError: If translation key is not found
        """
        return self._translation_manager.get_translation(
            self.task_name, key, language, **kwargs
        )

    @property
    def task_name(self) -> str:
        """Extract task name from module name or DATASET_NAME constant."""
        # Try to get from DATASET_NAME constant if available (preferred)
        if hasattr(self, "DATASET_NAME"):
            return self.DATASET_NAME

        # Fall back to extracting from module name
        module_name = self.__class__.__module__
        parts = module_name.split(".")
        if len(parts) >= 3:
            return parts[2]  # reasoning_gym.{category}.{dataset_name}
        return "unknown"
