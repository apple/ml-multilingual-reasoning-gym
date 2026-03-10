"""
Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""

"""Syllogism reasoning task generator"""

from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..coaching import BaseCurriculum, ScalarAttributeDefinition
from ..factory import register_dataset
from ..multilingual.base_classes import MultilingualProceduralDataset
from ..utils import StrEnum
from ..config import DatasetConfig

DATASET_NAME = "syllogisms"

class Quantifier(StrEnum):
    ALL = "All"
    NO = "No"
    SOME = "Some"
    SOME_NOT = "Some ... are not"

class Term:
    """Represents a categorical term used in syllogisms"""

    def __init__(self, name: str, plural: str):
        self.name = name
        self.plural = plural

    def __repr__(self) -> str:
        """Return string representation of the term"""
        return f"Term({self.name}, {self.plural})"

@dataclass
class SyllogismConfig(DatasetConfig):
    """Configuration for syllogism task generation"""

    # Control which quantifiers to use
    allow_all: bool = True
    allow_no: bool = True
    allow_some: bool = True
    allow_some_not: bool = True

    # Percentage of invalid examples if included (0.0 to 1.0)
    invalid_ratio: float = 0.3

    # Probability of generating inversion problems instead of syllogisms (0.0 to 1.0)
    inversion_probability: float = 0.3

    # Language support
    languages: list[str] | str = "en"
    language_weights: Optional[list[float]] = None

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert any(
            [self.allow_all, self.allow_no, self.allow_some, self.allow_some_not]
        ), "At least one quantifier type must be allowed"
        assert 0.0 <= self.invalid_ratio <= 1.0, "invalid_ratio must be between 0.0 and 1.0"
        assert 0.0 <= self.inversion_probability <= 1.0, "inversion_probability must be between 0.0 and 1.0"

class SyllogismDataset(MultilingualProceduralDataset):
    """Generates syllogism reasoning tasks"""

    # Term names for loading from translations
    TERM_NAMES = [
        # People
        "mortal", "human", "child", "adult", "parent", "grandparent",
        # Professions
        "philosopher", "student", "teacher", "doctor", "scientist", "artist",
        "musician", "writer", "programmer", "engineer", "lawyer", "chef",
        # Animals
        "animal", "mammal", "dog", "cat", "bird", "fish", "reptile", "insect",
        "butterfly", "bee", "ant", "spider", "horse", "elephant", "lion", 
        "tiger", "whale", "dolphin",
    ]

    def __init__(self, config: SyllogismConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _get_terms_for_language(self, language: str) -> list[Term]:
        """Load terms from translations for the specified language"""
        terms = []
        for term_name in self.TERM_NAMES:
            term_value = self._get_translation(f"term_{term_name}", language)
            terms.append(Term(term_value, term_value))  # Same value for both since we don't need plural distinction
        return terms

    def _get_allowed_quantifiers(self) -> list[Quantifier]:
        """Get list of allowed quantifiers based on config"""
        quantifiers = []
        if self.config.allow_all:
            quantifiers.append(Quantifier.ALL)
        if self.config.allow_no:
            quantifiers.append(Quantifier.NO)
        if self.config.allow_some:
            quantifiers.append(Quantifier.SOME)
        if self.config.allow_some_not:
            quantifiers.append(Quantifier.SOME_NOT)
        return quantifiers

    @staticmethod
    def _is_valid_syllogism(
        premise1: tuple[Quantifier, "Term", "Term"],
        premise2: tuple[Quantifier, "Term", "Term"],
        conclusion: tuple[Quantifier, "Term", "Term"],
    ) -> bool:
        """
        Checks whether a given syllogism is valid under classical (Aristotelian) rules,
        including the distribution rule:
        - If a term is distributed in the conclusion, it must be distributed
          in the premise where it appears as subject/predicate.
        """

        # --- 1) Extract data ---
        q1, p1_subj, p1_pred = premise1
        q2, p2_subj, p2_pred = premise2
        q3, c_subj, c_pred = conclusion

        negative_set = {Quantifier.NO, Quantifier.SOME_NOT}
        particular_set = {Quantifier.SOME, Quantifier.SOME_NOT}
        universal_set = {Quantifier.ALL, Quantifier.NO}

        # --- 2) Identify a unique middle term ---
        premise1_terms = {p1_subj, p1_pred}
        premise2_terms = {p2_subj, p2_pred}
        common_terms = premise1_terms.intersection(premise2_terms)

        if len(common_terms) != 1:
            return False
        middle_term = next(iter(common_terms))

        # Gather all terms => must be exactly 3 distinct terms
        all_terms = premise1_terms.union(premise2_terms)
        if len(all_terms) != 3:
            return False

        # The conclusion must use the other two terms (not the middle)
        other_two = all_terms - {middle_term}
        conclusion_terms = {c_subj, c_pred}
        if conclusion_terms != other_two:
            return False

        # --- 3) Identify which premise is major vs. minor ---
        def premise_contains(premise, term):
            return (premise[1] == term) or (premise[2] == term)

        if premise_contains(premise1, c_pred):
            major = premise1
            minor = premise2
        elif premise_contains(premise2, c_pred):
            major = premise2
            minor = premise1
        else:
            return False

        # The minor premise must contain the conclusion's subject
        if not premise_contains(minor, c_subj):
            return False

        # --- 4) Quick checks (traditional “no two negative,” etc.) ---
        if (q1 in negative_set) and (q2 in negative_set):
            return False
        if (q1 in particular_set) and (q2 in particular_set):
            return False
        if q3 in universal_set:
            if (q1 in particular_set) or (q2 in particular_set):
                return False
        if q3 in negative_set:
            if not ((q1 in negative_set) or (q2 in negative_set)):
                return False

        # --- 5) Distribution checks ---
        def distribution(q: Quantifier):
            if q == Quantifier.ALL:  # A
                return (True, False)
            elif q == Quantifier.NO:  # E
                return (True, True)
            elif q == Quantifier.SOME:  # I
                return (False, False)
            elif q == Quantifier.SOME_NOT:  # O
                return (False, True)
            else:
                raise ValueError(f"Unknown quantifier: {q}")

        # Major premise distribution
        q_major, major_subj, major_pred = major
        dist_major_subj, dist_major_pred = distribution(q_major)

        # Minor premise distribution
        q_minor, minor_subj, minor_pred = minor
        dist_minor_subj, dist_minor_pred = distribution(q_minor)

        # The middle term must be distributed in at least one of the premises.
        middle_is_dist_in_major = (
            (middle_term == major_subj and dist_major_subj) or
            (middle_term == major_pred and dist_major_pred)
        )
        
        middle_is_dist_in_minor = (
            (middle_term == minor_subj and dist_minor_subj) or
            (middle_term == minor_pred and dist_minor_pred)
        )

        if not (middle_is_dist_in_major or middle_is_dist_in_minor):
            return False # Invalid due to Undistributed Middle

        # Conclusion distribution
        dist_c_subj, dist_c_pred = distribution(q3)

        # If the conclusion's subject is distributed, check it in the minor premise
        if dist_c_subj:
            if c_subj == minor_subj:
                if not dist_minor_subj:
                    return False
            elif c_subj == minor_pred:
                if not dist_minor_pred:
                    return False

        # If the conclusion's predicate is distributed, check it in the major premise
        if dist_c_pred:
            if c_pred == major_subj:
                if not dist_major_subj:
                    return False
            elif c_pred == major_pred:
                if not dist_major_pred:
                    return False

        # If either premise is negative, the conclusion must be negative.
        if (q1 in negative_set) or (q2 in negative_set):
            if q3 not in negative_set:
                return False

        # If all checks pass, it's valid
        return True

    def _format_quantifier_statement(self, quantifier: Quantifier, subject: Term, predicate: Term, language: str) -> str:
        """Format a quantified statement in predicate logic notation using set membership"""
        
        # Get the term names for this language
        subject_name = subject.name
        predicate_name = predicate.name
        
        if quantifier == Quantifier.ALL:
            return f"∀x ∈ {subject_name}: x ∈ {predicate_name}"
        elif quantifier == Quantifier.NO:
            return f"∀x ∈ {subject_name}: x ∉ {predicate_name}"
        elif quantifier == Quantifier.SOME:
            return f"∃x ∈ {subject_name}: x ∈ {predicate_name}"
        elif quantifier == Quantifier.SOME_NOT:
            return f"∃x ∈ {subject_name}: x ∉ {predicate_name}"

    def _check_logical_equivalence(
        self, premise: tuple[Quantifier, Term, Term], conclusion: tuple[Quantifier, Term, Term]
    ) -> bool:
        """Check if a conclusion is logically equivalent to a premise"""
        p_quant, p_subj, p_pred = premise
        c_quant, c_subj, c_pred = conclusion

        # Direct inversion for universal negative
        if p_quant == Quantifier.NO:
            if c_quant == Quantifier.NO:
                return p_subj == c_pred and p_pred == c_subj
            return False

        # Particular inversion for universal affirmative
        if p_quant == Quantifier.ALL:
            if c_quant == Quantifier.SOME:
                return p_subj == c_pred and p_pred == c_subj
            return False

        # Rules for particular statements
        if p_quant == Quantifier.SOME:
            if c_quant == Quantifier.SOME:
                return p_subj == c_pred and p_pred == c_subj
            return False

        if p_quant == Quantifier.SOME_NOT:
            # Some A are not B does not imply Some B are not A
            return False

        return False

    def _generate_syllogism(self, rng: Random, idx: int, language: str) -> dict:
        """Generate a single syllogism problem"""
        # Get terms for the specified language
        terms = self._get_terms_for_language(language)
        
        # Select three different terms
        selected_terms = rng.sample(terms, 3)
        quantifiers = self._get_allowed_quantifiers()

        # Decide whether to generate a traditional syllogism or an inversion problem
        if rng.random() < self.config.inversion_probability:
            # Generate two premises, one will be used for inversion, the other as distractor
            quantifier1 = rng.choice(quantifiers)
            quantifier2 = rng.choice(quantifiers)
            term1, term2, term3 = selected_terms  # Use all three terms

            # Create two different premises
            premise1 = (quantifier1, term1, term2)
            premise2 = (quantifier2, term2, term3)

            # Format both premises
            premise1_text = self._format_quantifier_statement(premise1[0], premise1[1], premise1[2], language)
            premise2_text = self._format_quantifier_statement(premise2[0], premise2[1], premise2[2], language)

            # Randomly select which premise to use for inversion
            if rng.random() < 0.5:
                premise = premise1
                selected_premise_num = 1
            else:
                premise = premise2
                selected_premise_num = 2

            # Decide whether to generate a valid or invalid inversion
            target_valid = rng.random() > self.config.invalid_ratio

            # Get the quantifier and terms from the selected premise
            premise_quantifier, premise_term1, premise_term2 = premise

            if target_valid:
                # Generate valid inversions
                if premise_quantifier == Quantifier.NO:
                    conclusion = (premise_quantifier, premise_term2, premise_term1)  # No B are A
                elif premise_quantifier == Quantifier.ALL:
                    conclusion = (Quantifier.SOME, premise_term2, premise_term1)  # Some B are A
                elif premise_quantifier == Quantifier.SOME:
                    conclusion = (premise_quantifier, premise_term2, premise_term1)  # Some B are A
                else:  # SOME_NOT - try a different quantifier
                    new_quantifier = rng.choice([q for q in quantifiers if q != Quantifier.SOME_NOT])
                    # Update the premise with the new quantifier
                    premise = (new_quantifier, premise_term1, premise_term2)
                    premise_quantifier = new_quantifier  # Update the quantifier for conclusion generation
                    if selected_premise_num == 1:
                        premise1 = premise
                        premise1_text = self._format_quantifier_statement(premise[0], premise[1], premise[2], language)
                    else:
                        premise2 = premise
                        premise2_text = self._format_quantifier_statement(premise[0], premise[1], premise[2], language)

                    # Handle the new quantifier
                    if new_quantifier == Quantifier.NO:
                        conclusion = (new_quantifier, premise_term2, premise_term1)
                    elif new_quantifier == Quantifier.ALL:
                        conclusion = (Quantifier.SOME, premise_term2, premise_term1)
                    else:  # SOME
                        conclusion = (new_quantifier, premise_term2, premise_term1)
            else:
                # Generate invalid inversions by sampling from inappropriate quantifiers
                if premise_quantifier == Quantifier.NO:
                    # For NO statements, use ALL or SOME
                    conclusion = (rng.choice([Quantifier.ALL, Quantifier.SOME]), premise_term2, premise_term1)
                elif premise_quantifier == Quantifier.ALL:
                    # For ALL statements, use ALL or NO
                    conclusion = (rng.choice([Quantifier.ALL, Quantifier.NO]), premise_term2, premise_term1)
                elif premise_quantifier == Quantifier.SOME:
                    # For SOME statements, use ALL or NO
                    conclusion = (rng.choice([Quantifier.ALL, Quantifier.NO]), premise_term2, premise_term1)
                else:  # SOME_NOT
                    # For SOME_NOT statements, use any other quantifier
                    conclusion = (
                        rng.choice([q for q in quantifiers if q != Quantifier.SOME_NOT]),
                        premise_term2,
                        premise_term1,
                    )

            conclusion_text = self._format_quantifier_statement(conclusion[0], conclusion[1], conclusion[2], language)
            is_valid = self._check_logical_equivalence(premise, conclusion)

            question = (
                f"{self._get_translation('inversion_intro', language)}\n"
                f"1. {premise1_text}\n"
                f"2. {premise2_text}\n\n"
                f"{self._get_translation('inversion_question', language)}\n"
                f"{conclusion_text}?\n"
                f"{self._get_translation('answer_instruction', language)}"
            )

            return {
                "question": question,
                "answer": self._get_translation('yes', language) if is_valid else self._get_translation('no', language),
                "metadata": {
                    "source_dataset": DATASET_NAME,
                    "source_index": idx,
                    "premise1": premise1_text,
                    "premise2": premise2_text,
                    "selected_premise": selected_premise_num,
                    "conclusion": conclusion_text,
                    "is_valid": is_valid,
                    "type": "inversion",
                    "language": language,
                    "difficulty": {
                        "mock": 1
                    }
                },
            }

        # Traditional syllogism generation
        target_valid = rng.random() > self.config.invalid_ratio  # Invert ratio to match meaning
        max_attempts = 100
        attempts = 0

        while attempts < max_attempts:
            # Generate premises and conclusion
            premise1 = (rng.choice(quantifiers), selected_terms[0], selected_terms[1])
            premise2 = (rng.choice(quantifiers), selected_terms[1], selected_terms[2])
            conclusion = (rng.choice(quantifiers), selected_terms[0], selected_terms[2])

            # Check if validity matches target
            is_valid = self._is_valid_syllogism(premise1, premise2, conclusion)
            if is_valid == target_valid:
                break

            attempts += 1

        if attempts >= max_attempts:
            # If we couldn't find a matching syllogism, return a basic valid one
            premise1 = (Quantifier.ALL, selected_terms[0], selected_terms[1])
            premise2 = (Quantifier.ALL, selected_terms[1], selected_terms[2])
            conclusion = (Quantifier.ALL, selected_terms[0], selected_terms[2])
            is_valid = True

        # Format the syllogism as text
        premise1_text = self._format_quantifier_statement(premise1[0], premise1[1], premise1[2], language)
        premise2_text = self._format_quantifier_statement(premise2[0], premise2[1], premise2[2], language)
        conclusion_text = self._format_quantifier_statement(conclusion[0], conclusion[1], conclusion[2], language)

        question = (
            f"{self._get_translation('syllogism_intro', language)}\n"
            f"1. {premise1_text}\n"
            f"2. {premise2_text}\n\n"
            f"{self._get_translation('syllogism_question', language)}\n"
            f"{conclusion_text}?\n"
            f"{self._get_translation('answer_instruction', language)}"
        )

        return {
            "question": question,
            "answer": self._get_translation('yes', language) if is_valid else self._get_translation('no', language),
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "premise1": premise1_text,
                "premise2": premise2_text,
                "conclusion": conclusion_text,
                "is_valid": is_valid,
                "type": "syllogism",
                "language": language,
                "difficulty": {
                    "mock": 1
                }
            },
        }

    def _generate_item(self, idx: int, language: str) -> dict[str, Any]:
        """Generate a single syllogism task for the specified language"""
        rng = Random(self.seed + idx)
        return self._generate_syllogism(rng, idx, language)

class SyllogismCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(SyllogismCurriculum.__name__, SyllogismConfig)
        self._define_attributes(
            ScalarAttributeDefinition(
                name="allow_all",
                field_name="allow_all",
                levels=[True, True, True, True],
                description="Allow 'All' quantifier",
            ),
            ScalarAttributeDefinition(
                name="allow_no",
                field_name="allow_no",
                levels=[False, True, True, True],
                description="Allow 'No' quantifier",
            ),
            ScalarAttributeDefinition(
                name="allow_some",
                field_name="allow_some",
                levels=[False, False, True, True],
                description="Allow 'Some' quantifier",
            ),
            ScalarAttributeDefinition(
                name="allow_some_not",
                field_name="allow_some_not",
                levels=[False, False, False, True],
                description="Allow 'Some ... are not' quantifier",
            ),
        )

register_dataset(DATASET_NAME, SyllogismDataset, SyllogismConfig, SyllogismCurriculum)
