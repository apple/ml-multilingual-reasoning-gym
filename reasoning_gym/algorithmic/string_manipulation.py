"""
Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""

"""Manipulate a string according to a set of rules

https://github.com/yongchao98/CodeSteer-v1.0/blob/main/create_dataset/create_dataset_string_deletion_and_modification.py
"""

from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import register_dataset
from ..multilingual.base_classes import MultilingualProceduralDataset
from ..config import DatasetConfig

DATASET_NAME = "string_manipulation"

@dataclass
class StringManipulationConfig(DatasetConfig):
    """Configuration for String Insertion dataset generation"""

    min_string_length: int = 5  # Minimum string length
    max_string_length: int = 20  # Maximum string length
    min_num_rules: int = 3  # Minimum number of rules/transforms
    max_num_rules: int = 8  # Maximum number of rules/transforms
    languages: list[str] | str = "en"  # Languages to support
    language_weights: Optional[list[float]] = None  # Weights for language sampling

    def validate(self):
        """Validate configuration parameters"""
        assert 5 <= self.min_string_length, "Minimum string length should be at least 5"
        assert self.min_string_length <= self.max_string_length, "Minimum string length should be less than maximum"
        assert 3 <= self.min_num_rules, "Minimum number of rules should be at least 3"
        assert self.min_num_rules <= self.max_num_rules, "Minimum number of rules should be less than maximum"
        assert self.max_num_rules <= 20, "Maximum number of rules should be at most 20"

class StringManipulationDataset(MultilingualProceduralDataset):
    """Generates String Insertion exercises with configurable difficulty"""

    def __init__(self, config: StringManipulationConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)
        self.vocabulary = ["a", "b", "c"]
        self.rules = [
            (
                "rule_1",
                lambda s: ("ca" + s[2:], 1) if s.startswith("ab") else (s, 0),
            ),
            (
                "rule_2",
                lambda s: (s[:-2] + "cb", 2) if s.endswith("ac") else (s, 0),
            ),
            (
                "rule_3",
                lambda s: (s[2:] + "aa", 3) if s.startswith("bc") else (s, 0),
            ),
            (
                "rule_4",
                lambda s: (s[:-2], 4) if s.endswith("bb") else (s, 0),
            ),
            (
                "rule_5",
                lambda s: ("aa" + s[2:-1], 5) if s.startswith("cb") and len(s) > 1 else (s, 0),
            ),
            (
                "rule_6",
                lambda s: ("bb" + s[2:] + "c", 6) if s.startswith("ca") else (s, 0),
            ),
            (
                "rule_7",
                lambda s: ("a" + s[:-2] + "b", 7) if s.endswith("cc") else (s, 0),
            ),
            (
                "rule_8",
                lambda s: (s[1:], 8) if s.startswith("aa") else (s, 0),
            ),
            (
                "rule_9",
                lambda s: (s.replace("abc", "cab", 1), 9) if "abc" in s else (s, 0),
            ),
            (
                "rule_10",
                lambda s: (s.replace("bca", "", 1), 10) if "bca" in s else (s, 0),
            ),
            (
                "rule_11",
                lambda s: (s[:-2] + "ab", 11) if s.endswith("ba") else (s, 0),
            ),
            (
                "rule_12",
                lambda s: (s[2:], 12) if s.startswith("cc") else (s, 0),
            ),
            (
                "rule_13",
                lambda s: (s.replace("acb", "bca", 1), 13) if "acb" in s else (s, 0),
            ),
            (
                "rule_14",
                lambda s: (s[:-1], 14) if s.endswith("ca") and len(s) > 0 else (s, 0),
            ),
            (
                "rule_15",
                lambda s: (s[0] + s[2:], 15) if s.startswith("bb") and len(s) >= 2 else (s, 0),
            ),
            (
                "rule_16",
                lambda s: (s[:-2] + "cc", 16) if s.endswith("aa") else (s, 0),
            ),
            (
                "rule_17",
                lambda s: (s[:idx] + s[idx + 2 :], 17) if (idx := s.find("ca", 1)) != -1 else (s, 0),
            ),
            (
                "rule_18",
                lambda s: (s + "ab", 18) if (s.count("b") > 0 and s.count("b") % 2 == 0) else (s, 0),
            ),
            (
                "rule_19",
                lambda s: (s[: len(s) // 2] + s[len(s) // 2 + 1 :], 19) if len(s) > 15 else (s, 0),
            ),
            (
                "rule_20",
                lambda s: ("zz" + s[2:], 20) if s.startswith("ac") else (s, 0),
            ),
        ]

    def _apply_rule(self, string: str, selected_rules: list[tuple[str, callable]]) -> tuple[str, int]:
        """
        Apply the first applicable rule from the list of selected rules.
        Returns a tuple containing the modified string and the rule index (1-based) that was applied.
        If no rule is applicable, returns (s, 0).
        """
        for _, rule_fn in selected_rules:
            new_string, op_idx = rule_fn(string)
            if op_idx != 0:
                return new_string, op_idx
        return string, 0

    def _get_all_transforms(self, string: str, selected_rules: list[tuple[str, callable]]) -> list[str]:
        """
        Repeatedly apply transformation rules to a string until no further transformations can be performed,
        or a state is repeated. If a state is repeated, the process is terminated, and the state is not added to the list.
        Returns a list of string states from the initial string to the final state (i.e. the desired answer).
        """
        states = [string]
        while True:
            new_string, op_idx = self._apply_rule(states[-1], selected_rules)
            if op_idx == 0 or new_string in states:
                break
            states.append(new_string)
        return states

    def _generate_item(self, idx: int, language: str) -> dict[str, Any]:
        """Generate a single String Manipulation question in the specified language"""
        rng = Random(self.seed + idx)

        string_length = rng.randint(self.config.min_string_length, self.config.max_string_length)
        string = "".join(rng.choice(self.vocabulary) for _ in range(string_length))

        num_rules = rng.randint(self.config.min_num_rules, self.config.max_num_rules)
        selected_rules = rng.sample(self.rules, num_rules)
        
        # Build rules string with translations
        rules_lines = []
        for i, (rule_key, _) in enumerate(selected_rules):
            rule_text = self._get_translation(rule_key, language)
            rules_lines.append(f"{i+1}. {rule_text}")
        rules_str = "\n".join(rules_lines)

        states = self._get_all_transforms(string, selected_rules)
        answer = states[-1]

        question = self._get_translation("question_template", language, 
                                        rules=rules_str, 
                                        string=string)

        return {
            "question": question,
            "answer": str(answer),
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "string": string,
                "solution": answer,
                "states": states,
                "selected_rules": [rule for rule, _ in selected_rules],
                "string_length": string_length,
                "num_rules": num_rules,
                "language": language,
                "difficulty": {
                    "string_length": (self.config.min_string_length, self.config.max_string_length),
                    "num_rules": (self.config.min_num_rules, self.config.max_num_rules),
                },
            },
        }

class StringManipulationCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(StringManipulationCurriculum.__name__, StringManipulationConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="string_length",
                levels=[10, 50, 100, 500],
                description="Length of the string",
                lower_field_name="min_string_length",
                upper_field_name="max_string_length",
                ensure_interval=True,
            ),
            RangeAttributeDefinition(
                name="num_rules",
                levels=[3, 5, 10, 15, 20],
                description="Number of rules to apply",
                lower_field_name="min_num_rules",
                upper_field_name="max_num_rules",
                ensure_interval=True,
            ),
        )

register_dataset(DATASET_NAME, StringManipulationDataset, StringManipulationConfig, StringManipulationCurriculum)
