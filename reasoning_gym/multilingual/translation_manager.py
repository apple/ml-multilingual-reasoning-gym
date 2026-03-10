"""
Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""

"""Translation manager for multilingual dataset support."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional


class TranslationManager:
    """Manages translations for reasoning-gym tasks using JSON templates."""

    def __init__(self):
        """Initialize translation manager."""
        self.translations_dir = Path(__file__).parent / "translations"

    def get_translation(self, task_name: str, key: str, language: str, **kwargs) -> str:
        """Get translation from JSON template.

        Args:
            task_name: Name of the task
            key: Translation key
            language: Language code
            **kwargs: Parameters for template formatting

        Returns:
            Formatted translation text

        Raises:
            KeyError: If translation key is not found
            FileNotFoundError: If template file is not found
        """
        templates = self._load_json_templates(task_name, language)
        if key not in templates:
            raise KeyError(
                f"Translation key '{key}' not found for task '{task_name}' in language '{language}'"
            )

        try:
            return templates[key].format(**kwargs)
        except KeyError as e:
            raise KeyError(f"Missing template parameter for '{key}': {e}")

    @lru_cache(maxsize=256)
    def _load_json_templates(self, task_name: str, language: str) -> Dict[str, str]:
        """Load translated JSON templates.

        Args:
            task_name: Name of the task
            language: Language code

        Returns:
            Dictionary of translation templates

        Raises:
            FileNotFoundError: If JSON template file is not found
        """
        # Auto-discover task group
        task_group = self._discover_task_group(task_name)
        if not task_group:
            raise FileNotFoundError(f"Task '{task_name}' not found in any task group")

        json_file = self.translations_dir / task_group / task_name / f"{language}.json"
        if not json_file.exists():
            raise FileNotFoundError(
                f"JSON template for task '{task_name}' not available in language '{language}'"
            )

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            raise FileNotFoundError(
                f"Failed to load JSON template for task '{task_name}' in language '{language}': {e}"
            )

    @lru_cache(maxsize=128)
    def _discover_task_group(self, task_name: str) -> Optional[str]:
        """Auto-discover which group a task belongs to by scanning the filesystem.

        Args:
            task_name: Name of the task

        Returns:
            Group name if found, None otherwise
        """
        for group_dir in self.translations_dir.iterdir():
            if group_dir.is_dir():
                task_dir = group_dir / task_name
                if task_dir.exists() and task_dir.is_dir():
                    return group_dir.name
        return None

    def get_available_languages(self, task_name: str) -> List[str]:
        """Get available languages for a specific task.

        Args:
            task_name: Name of the task

        Returns:
            Sorted list of available language codes
        """
        return sorted(self._get_json_languages(task_name))

    def _get_json_languages(self, task_name: str) -> List[str]:
        """Get available languages from JSON files for a task.

        Args:
            task_name: Name of the task

        Returns:
            List of available language codes from JSON files
        """
        task_group = self._discover_task_group(task_name)
        if not task_group:
            return []

        task_dir = self.translations_dir / task_group / task_name
        if not task_dir.exists():
            return []

        languages = []
        for json_file in task_dir.glob("*.json"):
            languages.append(json_file.stem)
        return languages

    def get_available_tasks(self) -> Dict[str, List[str]]:
        """Get all available tasks organized by group.

        Returns:
            Dictionary mapping group names to list of task names
        """
        tasks_by_group = {}

        if not self.translations_dir.exists():
            return tasks_by_group

        for group_dir in self.translations_dir.iterdir():
            if group_dir.is_dir():
                group_name = group_dir.name
                tasks = []
                for task_dir in group_dir.iterdir():
                    if task_dir.is_dir():
                        tasks.append(task_dir.name)
                if tasks:  # Only include groups that have tasks
                    tasks_by_group[group_name] = sorted(tasks)
        return tasks_by_group
