"""
Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""

"""Tests for multilingual base classes."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from reasoning_gym.config import DatasetConfig
from reasoning_gym.multilingual.base_classes import MultilingualProceduralDataset


class MockMultilingualDataset(MultilingualProceduralDataset):
    """Mock implementation of multilingual dataset for testing."""

    DATASET_NAME = "mock_dataset"

    def _generate_item(self, idx: int, language: str) -> dict:
        return {
            "question": self._get_translation("greeting", language, name="World"),
            "answer": "42",
            "metadata": {
                "language": language,
                "index": idx
            }
        }


class TestMultilingualProceduralDataset:
    """Test cases for multilingual dataset base class."""

    def _create_translation_files(self, temp_dir: Path, task_name: str, languages_dict: dict):
        """Helper to create translation JSON files.

        Args:
            temp_dir: Temporary directory path
            task_name: Name of the task
            languages_dict: Dict mapping language codes to translation dicts
        """
        task_dir = temp_dir / "test_group" / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        for lang, translations in languages_dict.items():
            lang_file = task_dir / f"{lang}.json"
            with open(lang_file, 'w', encoding='utf-8') as f:
                json.dump(translations, f)

    def _mock_translation_manager_init(self, translations_dir):
        """Create a mock __init__ for TranslationManager that uses the given directory."""
        def mock_init(self):
            self.translations_dir = Path(translations_dir)
        return mock_init

    def test_single_language_config(self):
        """Test dataset with single language configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            translations_dir = Path(temp_dir)

            # Create English translation
            self._create_translation_files(
                translations_dir,
                "mock_dataset",
                {"es": {"greeting": "Hola, {name}!"}}
            )

            config = DatasetConfig(size=5, seed=42, languages="es")

            # Patch TranslationManager to use temp directory
            with patch('reasoning_gym.multilingual.translation_manager.TranslationManager.__init__',
                      self._mock_translation_manager_init(translations_dir)):
                dataset = MockMultilingualDataset(config)

                assert dataset.languages == ["es"]

                # All items should be in English
                for i in range(3):
                    item = dataset[i]
                    assert item["metadata"]["language"] == "es"
                    assert "Hola, World!" in item["question"]

    def test_multiple_languages_config(self):
        """Test dataset with multiple languages."""
        with tempfile.TemporaryDirectory() as temp_dir:
            translations_dir = Path(temp_dir)

            # Create English and Spanish translations
            self._create_translation_files(
                translations_dir,
                "mock_dataset",
                {
                    "en": {"greeting": "Hello, {name}!"},
                    "es": {"greeting": "¡Hola, {name}!"}
                }
            )

            config = DatasetConfig(
                size=10,
                seed=42,
                languages=["en", "es"],
                language_weights=[1.0, 1.0]
            )

            # Patch TranslationManager to use temp directory
            with patch('reasoning_gym.multilingual.translation_manager.TranslationManager.__init__',
                      self._mock_translation_manager_init(translations_dir)):
                dataset = MockMultilingualDataset(config)

                assert set(dataset.languages) == {"en", "es"}

                # Check that we get both languages
                languages_seen = set()
                for i in range(10):
                    item = dataset[i]
                    lang = item["metadata"]["language"]
                    languages_seen.add(lang)

                    if lang == "en":
                        assert "Hello, World!" in item["question"]
                    elif lang == "es":
                        assert "¡Hola, World!" in item["question"]

                # Should have seen both languages with enough samples
                assert len(languages_seen) == 2

    def test_deterministic_language_selection(self):
        """Test that language selection is deterministic for datasets with same config.seed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            translations_dir = Path(temp_dir)

            # Create English and Spanish translations
            self._create_translation_files(
                translations_dir,
                "mock_dataset",
                {
                    "en": {"greeting": "Hello, {name}!"},
                    "es": {"greeting": "¡Hola, {name}!"}
                }
            )

            # Patch TranslationManager to use temp directory
            with patch('reasoning_gym.multilingual.translation_manager.TranslationManager.__init__',
                      self._mock_translation_manager_init(translations_dir)):
                # Test within single dataset (should always work)
                config = DatasetConfig(size=10, seed=42, languages=["en", "es"])
                dataset = MockMultilingualDataset(config)

                # Get same item multiple times - should be deterministic
                item1_first = dataset[1]
                item1_second = dataset[1]
                item5_first = dataset[5]
                item5_second = dataset[5]

                assert item1_first["metadata"]["language"] == item1_second["metadata"]["language"]
                assert item5_first["metadata"]["language"] == item5_second["metadata"]["language"]

                # Test across datasets with same config.seed
                config1 = DatasetConfig(size=5, seed=42, languages=["en", "es"])
                config2 = DatasetConfig(size=5, seed=42, languages=["en", "es"])
                dataset1 = MockMultilingualDataset(config1)
                dataset2 = MockMultilingualDataset(config2)

                for i in range(5):
                    item1 = dataset1[i]
                    item2 = dataset2[i]
                    assert item1["metadata"]["language"] == item2["metadata"]["language"]

    def test_language_weights(self):
        """Test that language weights affect distribution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            translations_dir = Path(temp_dir)

            # Create English and Spanish translations
            self._create_translation_files(
                translations_dir,
                "mock_dataset",
                {
                    "en": {"greeting": "Hello, {name}!"},
                    "es": {"greeting": "¡Hola, {name}!"}
                }
            )

            # Heavy weight for English
            config = DatasetConfig(
                size=100,
                seed=42,
                languages=["en", "es"],
                language_weights=[10.0, 1.0]
            )

            # Patch TranslationManager to use temp directory
            with patch('reasoning_gym.multilingual.translation_manager.TranslationManager.__init__',
                      self._mock_translation_manager_init(translations_dir)):
                dataset = MockMultilingualDataset(config)

                en_count = 0
                es_count = 0

                for i in range(100):
                    item = dataset[i]
                    if item["metadata"]["language"] == "en":
                        en_count += 1
                    else:
                        es_count += 1

                # English should be much more frequent (rough check)
                assert en_count > es_count * 5

    def test_missing_translation_validation(self):
        """Test validation fails when translations are missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            translations_dir = Path(temp_dir)

            # Only create English translation
            self._create_translation_files(
                translations_dir,
                "mock_dataset",
                {"en": {"greeting": "Hello, {name}!"}}
            )

            # Try to create dataset with Spanish (not available)
            config = DatasetConfig(size=5, seed=42, languages=["en", "es"])

            # Patch TranslationManager to use temp directory
            with patch('reasoning_gym.multilingual.translation_manager.TranslationManager.__init__',
                      self._mock_translation_manager_init(translations_dir)):
                with pytest.raises(FileNotFoundError, match="not available in languages: \\['es'\\]"):
                    MockMultilingualDataset(config)

    def test_task_name_from_dataset_name(self):
        """Test that task_name property works correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            translations_dir = Path(temp_dir)

            # Need to create translation because __init__ validates translations exist
            self._create_translation_files(
                translations_dir,
                "mock_dataset",
                {"en": {"greeting": "Hello, {name}!"}}
            )

            config = DatasetConfig(size=1, seed=42, languages=["en"])

            # Patch TranslationManager to use temp directory
            with patch('reasoning_gym.multilingual.translation_manager.TranslationManager.__init__',
                      self._mock_translation_manager_init(translations_dir)):
                dataset = MockMultilingualDataset(config)

                assert dataset.task_name == "mock_dataset"
