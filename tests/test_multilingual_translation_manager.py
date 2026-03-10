"""
Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""

"""Tests for multilingual translation manager."""

import pytest
import json
import tempfile
from pathlib import Path
from reasoning_gym.multilingual.translation_manager import TranslationManager


class TestTranslationManager:
    """Test cases for the translation manager."""

    def test_json_only_translation(self):
        """Test translation using only JSON templates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            translations_dir = Path(temp_dir)
            task_dir = translations_dir / "arithmetic" / "test_task"
            task_dir.mkdir(parents=True)

            en_file = task_dir / "en.json"
            en_data = {
                "greeting": "Hello {name}!",
                "farewell": "Goodbye {name}!"
            }
            with open(en_file, 'w') as f:
                json.dump(en_data, f)

            manager = TranslationManager()
            manager.translations_dir = translations_dir

            result = manager.get_translation("test_task", "greeting", "en", name="Alice")
            assert result == "Hello Alice!"

            result = manager.get_translation("test_task", "farewell", "en", name="Bob")
            assert result == "Goodbye Bob!"

    def test_missing_translation_error(self):
        """Test error when JSON template doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            translations_dir = Path(temp_dir)
            manager = TranslationManager()
            manager.translations_dir = translations_dir

            with pytest.raises(FileNotFoundError):
                manager.get_translation("nonexistent_task", "greeting", "en")

    def test_missing_key_error(self):
        """Test error when JSON exists but key is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            translations_dir = Path(temp_dir)
            task_dir = translations_dir / "arithmetic" / "test_task"
            task_dir.mkdir(parents=True)

            en_file = task_dir / "en.json"
            en_data = {"greeting": "Hello!"}
            with open(en_file, 'w') as f:
                json.dump(en_data, f)

            manager = TranslationManager()
            manager.translations_dir = translations_dir

            with pytest.raises(KeyError):
                manager.get_translation("test_task", "missing_key", "en")

    def test_get_available_languages(self):
        """Test getting available languages from JSON files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            translations_dir = Path(temp_dir)
            task_dir = translations_dir / "arithmetic" / "test_task"
            task_dir.mkdir(parents=True)

            # Create JSON for English, Spanish and French
            en_file = task_dir / "en.json"
            es_file = task_dir / "es.json"
            fr_file = task_dir / "fr.json"

            with open(en_file, 'w') as f:
                json.dump({"greeting": "Hello!"}, f)
            with open(es_file, 'w') as f:
                json.dump({"greeting": "¡Hola!"}, f)
            with open(fr_file, 'w') as f:
                json.dump({"greeting": "Bonjour!"}, f)

            manager = TranslationManager()
            manager.translations_dir = translations_dir

            languages = manager.get_available_languages("test_task")
            assert set(languages) == {"en", "es", "fr"}

    def test_get_available_tasks(self):
        """Test getting available tasks organized by group."""
        with tempfile.TemporaryDirectory() as temp_dir:
            translations_dir = Path(temp_dir)

            # Create tasks in different groups
            (translations_dir / "arithmetic" / "task1").mkdir(parents=True)
            (translations_dir / "arithmetic" / "task2").mkdir(parents=True)
            (translations_dir / "algebra" / "task3").mkdir(parents=True)

            manager = TranslationManager()
            manager.translations_dir = translations_dir

            tasks = manager.get_available_tasks()
            assert tasks == {
                "algebra": ["task3"],
                "arithmetic": ["task1", "task2"]
            }