"""
Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""

"""Multilingual support for Reasoning Gym datasets."""

from .translation_manager import TranslationManager
from .base_classes import MultilingualProceduralDataset

__all__ = [
    "TranslationManager",
    "MultilingualProceduralDataset",
]
