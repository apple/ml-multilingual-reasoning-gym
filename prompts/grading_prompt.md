# Translation Quality Grading Prompt for Reasoning Gym Tasks

<task>
You are evaluating the quality of **template translations** used to procedurally generate math, logic and reasoning problems. Based on the context and examples provided, determine if the translation quality is excellent or needs further refinement.
</task>

<original_english_template>
```json
{ENGLISH_JSON_CONTENT}
```
</original_english_template>

<current_translated_template>
```json
{TARGET_LANGUAGE_JSON_CONTENT}
```
</current_translated_template>

<implementation_context>
```python
{PYTHON_FILE_CONTENT}
```
</implementation_context>

<template_explanation>
The templates above are used in the Python code to generate thousands of different problems by:
1. Filling placeholders like {{expression}} with mathematical expressions
2. Combining multiple templates for variety
3. Creating coherent problem statements that feel natural to native speakers

In the following, we give a few examples from filling in the placeholders with concrete values, showing both the English original and the target language translation.
</template_explanation>

<filled_out_template_examples>
{FILLEDOUT_EXAMPLES}
</filled_out_template_examples>

<human_feedback>
{HUMAN_FEEDBACK}
</human_feedback>

<complete_few_shot_examples>

<example_1>
<example_name>Count Bits Dataset</example_name>

<original_template>
```json
{
    "question_template": "How many 1 bits are there in the binary representation of the number {number}?"
}
```
</original_template>

<translated_template>
```json
{
    "question_template": "Wie viele 1-Bits gibt es in der binären Darstellung der Zahl {number}?"
}
```
</translated_template>

<implementation_context>
```python
"""Count number of 1 bits in a number."""

from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import register_dataset
from ..multilingual.base_classes import MultilingualProceduralDataset
from ..config import DatasetConfig

DATASET_NAME = "count_bits"

@dataclass
class CountBitsConfig(DatasetConfig):
    """Configuration for Count Bits dataset generation"""

    min_n: int = 1  # Minimum number to consider
    max_n: int = 2**31 - 1  # Maximum number to consider
    languages: list[str] | str = "en"
    language_weights: Optional[list[float]] = None

    def validate(self):
        """Validate configuration parameters"""
        assert 1 <= self.min_n <= self.max_n, "min_n must be between 1 and max_n"

class CountBitsDataset(MultilingualProceduralDataset):
    """Generates Count Bits exercises with configurable difficulty"""

    def __init__(self, config: CountBitsConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _generate_item(self, idx: int, language: str) -> dict[str, Any]:
        """Generate a single Count Bits question in the specified language"""
        rng = Random(self.seed + idx)

        number = rng.randint(self.config.min_n, self.config.max_n)
        binary = bin(number)[2:]
        answer = binary.count("1")

        question = self._get_translation("question_template", language, number=number)

        return {
            "question": question,
            "answer": str(answer),
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "number": number,
                "solution": answer,
                "binary": binary,
                "n": number,
                "difficulty": {
                    "n": (self.config.min_n, self.config.max_n),
                },
                "language": language,
            },
        }
```
</implementation_context>

<filled_out_template_examples>

<example_1>
<english>
<question>How many 1 bits are there in the binary representation of the number 1373158607?</question>
<answer>18</answer>
</english>
<german>
<question>Wie viele 1-Bits gibt es in der binären Darstellung der Zahl 1373158607?</question>
<answer>18</answer>
</german>
</example_1>

<example_2>
<english>
<question>How many 1 bits are there in the binary representation of the number 82789451?</question>
<answer>14</answer>
</english>
<german>
<question>Wie viele 1-Bits gibt es in der binären Darstellung der Zahl 82789451?</question>
<answer>14</answer>
</german>
</example_2>

<example_3>
<english>
<question>How many 1 bits are there in the binary representation of the number 877324117?</question>
<answer>16</answer>
</english>
<german>
<question>Wie viele 1-Bits gibt es in der binären Darstellung der Zahl 877324117?</question>
<answer>16</answer>
</german>
</example_3>

<example_4>
<english>
<question>How many 1 bits are there in the binary representation of the number 583848003?</question>
<answer>12</answer>
</english>
<german>
<question>Wie viele 1-Bits gibt es in der binären Darstellung der Zahl 583848003?</question>
<answer>12</answer>
</german>
</example_4>

<example_5>
<english>
<question>How many 1 bits are there in the binary representation of the number 1907541172?</question>
<answer>15</answer>
</english>
<german>
<question>Wie viele 1-Bits gibt es in der binären Darstellung der Zahl 1907541172?</question>
<answer>15</answer>
</german>
</example_5>
</filled_out_template_examples>

<example_feedback_output>
```json
{
  "status": "refinement_needed",
  "translation_quality_is_excellent": false,
  "concrete_issues": "The translation of '1-Bits' sounds too literal - in German, it would be more natural to say 'Einsen' (ones). The phrase 'binären Darstellung' could be more naturally expressed as the compound noun 'Binärdarstellung'. Mathematical terminology should follow German conventions more consistently.",
  "templating_issues": "",
  "other_issues": ""
}
```
</example_feedback_output>
</example_1>

<example_2>
<example_name>List Functions Dataset</example_name>

<original_template>
```json
{
    "prompt_template": "You are an expert at inductive reasoning. Generate an output corresponding to the given input.\\nThe output is generated by applying the same rule that maps input to output for the examples provided. Your answer should be a list of element/elements\\nExamples:\\n{examples}\\n\\nInput: {input}\\nOutput:\\n",
    "example_format": "Input {index}: {input_key}\\nOutput {index}: {value}\\n"
}
```
</original_template>

<translated_template>
```json
{
    "prompt_template": "Du bist ein Experte für induktives Schließen. Generiere eine Ausgabe entsprechend der gegebenen Eingabe.\\nDie Ausgabe wird generiert, indem dieselbe Regel angewendet wird, die in den bereitgestellten Beispielen die Eingabe auf die Ausgabe abbildet. Deine Antwort sollte eine Liste von Elementen sein.\\nBeispiele:\\n{examples}\\n\\nEingabe: {input}\\nAusgabe:\\n",
    "example_format": "Eingabe {index}: {input_key}\\nAusgabe {index}: {value}\\n"
}
```
</translated_template>

<implementation_context>
```python
"""List functions generators"""

from dataclasses import dataclass
from random import Random
from typing import Any, Callable, Optional

from reasoning_gym.factory import register_dataset
from reasoning_gym.multilingual.base_classes import MultilingualProceduralDataset
from reasoning_gym.config import DatasetConfig

DATASET_NAME = "list_functions"

@dataclass
class ListFunctionsDatasetConfig(DatasetConfig):
    """Configuration for List function generators."""

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
        self.task_indices = Random(self.seed).choices(tasks, k=self.size)

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
        """Generate a single induction-based list function dataset for the given language"""
        rng = Random(self.seed + idx)
        generator_idx = self.task_indices[idx]
        generator = self.generators[generator_idx]
        examples = generator(rng)
        entry = examples.popitem()
        input = entry[0]
        output = entry[1]
        formatted_examples = ""
        for index, input_key in enumerate(examples):
            formatted_examples += self._get_translation("example_format", language, 
                                                      index=index + 1, input_key=input_key, value=examples[input_key])
        question = self._get_translation("prompt_template", language, 
                                       examples=formatted_examples, input=input)
        return {
            "question": question,
            "answer": output,
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "language": language,
            },
        }
```
</implementation_context>

<filled_out_template_examples>

<example_1>
<english>
<question>
You are an expert at inductive reasoning. Generate an output corresponding to the given input.
The output is generated by applying the same rule that maps input to output for the examples provided. Your answer should be a list of element/elements
Examples:
Input 1: [4, 95, 36, 32]
Output 1: [4, 32, 36, 95]
Input 2: [18, 95, 14, 87, 95, 70]
Output 2: [14, 18, 70, 87, 95, 95]
Input 3: [76, 55, 5, 4]
Output 3: [4, 5, 55, 76]
Input 4: [28, 30, 65, 78]
Output 4: [28, 30, 65, 78]


Input: [72, 26, 92]
Output:
</question>
<answer>[26, 72, 92]</answer>
</english>
<german>
<question>
Du bist ein Experte für induktives Schließen. Generiere eine Ausgabe entsprechend der gegebenen Eingabe.
Die Ausgabe wird generiert, indem dieselbe Regel angewendet wird, die in den bereitgestellten Beispielen die Eingabe auf die Ausgabe abbildet. Deine Antwort sollte eine Liste von Elementen sein.
Beispiele:
Eingabe 1: [4, 95, 36, 32]
Ausgabe 1: [4, 32, 36, 95]
Eingabe 2: [18, 95, 14, 87, 95, 70]
Ausgabe 2: [14, 18, 70, 87, 95, 95]
Eingabe 3: [76, 55, 5, 4]
Ausgabe 3: [4, 5, 55, 76]
Eingabe 4: [28, 30, 65, 78]
Ausgabe 4: [28, 30, 65, 78]


Eingabe: [72, 26, 92]
Ausgabe:
</question>
<answer>[26, 72, 92]</answer>
</german>
</example_1>

<example_2>
<english>
<question>
You are an expert at inductive reasoning. Generate an output corresponding to the given input.
The output is generated by applying the same rule that maps input to output for the examples provided. Your answer should be a list of element/elements
Examples:
Input 1: [37, 90, 98]
Output 1: [37, 90, 98]
Input 2: [60, 48, 86, 90, 13]
Output 2: [60, 48, 86, 90, 13]
Input 3: [77, 64, 78, 3, 66, 56, 74, 48, 80, 71]
Output 3: [77, 64, 78, 3, 66, 56, 74, 48, 80, 71]
Input 4: [51, 23, 8, 14, 16, 49, 20, 13, 21]
Output 4: [51, 23, 8, 14, 16, 49, 20, 13, 21]


Input: [17, 99, 50, 77, 65, 35, 74, 24, 49, 9]
Output:
</question>
<answer>[17, 99, 50, 77, 65, 35, 74, 24, 49, 9]</answer>
</english>
<german>
<question>
Du bist ein Experte für induktives Schließen. Generiere eine Ausgabe entsprechend der gegebenen Eingabe.
Die Ausgabe wird generiert, indem dieselbe Regel angewendet wird, die in den bereitgestellten Beispielen die Eingabe auf die Ausgabe abbildet. Deine Antwort sollte eine Liste von Elementen sein.
Beispiele:
Eingabe 1: [37, 90, 98]
Ausgabe 1: [37, 90, 98]
Eingabe 2: [60, 48, 86, 90, 13]
Ausgabe 2: [60, 48, 86, 90, 13]
Eingabe 3: [77, 64, 78, 3, 66, 56, 74, 48, 80, 71]
Ausgabe 3: [77, 64, 78, 3, 66, 56, 74, 48, 80, 71]
Eingabe 4: [51, 23, 8, 14, 16, 49, 20, 13, 21]
Ausgabe 4: [51, 23, 8, 14, 16, 49, 20, 13, 21]


Eingabe: [17, 99, 50, 77, 65, 35, 74, 24, 49, 9]
Ausgabe:
</question>
<answer>[17, 99, 50, 77, 65, 35, 74, 24, 49, 9]</answer>
</german>
</example_2>

<example_3>
<english>
<question>
You are an expert at inductive reasoning. Generate an output corresponding to the given input.
The output is generated by applying the same rule that maps input to output for the examples provided. Your answer should be a list of element/elements
Examples:
Input 1: [4, 29, 49, 15, 90, 23, 38, 5, 67, 5, 70]
Output 1: [2]
Input 2: [37, 66, 21, 15, 44, 46, 80, 10]
Output 2: [0]
Input 3: [13, 45, 5, 5, 5, 50, 5]
Output 3: [4]
Input 4: [88, 6, 87]
Output 4: [0]


Input: [59, 5, 81, 5, 20, 5, 61, 76, 48, 70, 5, 30]
Output:
</question>
<answer>[4]</answer>
</english>
<german>
<question>
Du bist ein Experte für induktives Schließen. Generiere eine Ausgabe entsprechend der gegebenen Eingabe.
Die Ausgabe wird generiert, indem dieselbe Regel angewendet wird, die in den bereitgestellten Beispielen die Eingabe auf die Ausgabe abbildet. Deine Antwort sollte eine Liste von Elementen sein.
Beispiele:
Eingabe 1: [4, 29, 49, 15, 90, 23, 38, 5, 67, 5, 70]
Ausgabe 1: [2]
Eingabe 2: [37, 66, 21, 15, 44, 46, 80, 10]
Ausgabe 2: [0]
Eingabe 3: [13, 45, 5, 5, 5, 50, 5]
Ausgabe 3: [4]
Eingabe 4: [88, 6, 87]
Ausgabe 4: [0]


Eingabe: [59, 5, 81, 5, 20, 5, 61, 76, 48, 70, 5, 30]
Ausgabe:
</question>
<answer>[4]</answer>
</german>
</example_3>
</filled_out_template_examples>

<example_feedback_output>
```json
{
  "status": "refinement_needed",
  "translation_quality_is_excellent": false,
  "concrete_issues": "Several translation quality issues identified: 1) The term 'induktives Schließen' sounds unnatural in German - simply 'Induktion' would be more appropriate. 2) The translation 'Generiere' is too literal - 'Erstelle' or 'Erzeuge' would sound more natural. 3) The term 'bereitgestellten' is suboptimal - 'gegebenen' would be better German style. 4) The phrase 'Deine Antwort sollte eine Liste von Elementen sein' sounds stilted - 'Gib eine Liste von Elementen an' would be more natural and direct.",
  "templating_issues": "",
  "other_issues": ""
}
```
</example_feedback_output>
</example_2>
</complete_few_shot_examples>

<quality_assessment_criteria>

<primary_question>Do the translated templates generate natural, fluent mathematical problems in the target language? Judge the templates by their OUTPUT (the generated problems above), not by how literally they translate the English templates. A template that seems awkward but generates natural problems is EXCELLENT. A literal translation that generates stilted problems is POOR.</primary_question>

<critical_note>Pay particular attention to grammatical completeness. Templates must generate problems with proper grammatical structure, including all required particles, auxiliaries, and connecting words that native speakers expect in instructional contexts.</critical_note>

<translation_quality_issues>
The following is a non-exhaustive list of potential translation quality issues:
- Untranslated English artifacts in generated problems
- Mathematical incorrectness or conceptual errors in generated problems
- Template placeholder corruption (`{expression}` changed or translated)
- Generated problems are incomprehensible or significantly unnatural
- Generated problems sound unnatural or stilted in target language
- Mathematical terminology inconsistency across generated problems
- Cultural inappropriateness in mathematical expressions or formats
- Generated problems use inappropriate formality level for educational content
- Grammatically incomplete or incorrect constructions in generated problems
- Missing required grammatical elements (particles, auxiliaries, connectors)
- Incorrect verb forms or tense usage for instructional content
- Syntactic errors that native speakers would immediately notice
- Minor grammatical imperfections in generated problems
- Suboptimal word choices that don't affect problem comprehension
- Minor style inconsistencies across generated problems
- Pay special attention to grammatical completeness - ensure all required grammatical elements are present
- Verify that instructional verbs are used correctly according to target language grammar rules
- Check that mathematical instruction phrasing follows natural patterns in the target language
</translation_quality_issues>

<important_notes>
- If Human Feedback is provided, first assess which feedback points are still applicable to the current translation, then verify that the translation appropriately addresses those applicable feedback points and carry over any unaddressed applicable issues to the current assessment
- If Previous LLM Feedback is provided, verify which points have been addressed and carry over any unaddressed issues to the current assessment
- Be extremely critical when analyzing the translation quality and ensure a high bar for marking the translations as excellent
- Pay special attention to whether specific applicable human feedback has been acknowledged in the translation - translations that ignore clear applicable feedback should be marked as needing refinement
- When human feedback points have been resolved in the current translation, do not penalize the translation for those resolved issues
</important_notes>

<quality_thresholds>
- **Excellent**: Templates generate natural, fluent, mathematically correct problems ready for educational use
- **Needs Improvement**: Any issues fixable by better translation of the templates exist
</quality_thresholds>

<additional_issue_categories>
**A. Templating Issues** (Template structure doesn't work in target language):
- Template constructions that work in English but generate awkward problems in target language
- Placeholder positioning that creates unnatural word order in generated problems
- Cultural/linguistic patterns that need adaptation (e.g., enumeration styles, number formatting)

**B. Other Issues**
- Any other issues that don't stem from poor translation or templating issues
</additional_issue_categories>

<assessment_examples>
**EXCELLENT Template**: Generates "Berechne 2+3." (natural German) even if template "Berechne {expression}." seems simple

**POOR Template**: Generates "Was ist von 2+3?" (unnatural German) even if template translation seems literally correct
</assessment_examples>

<task_instructions>
Analyze the provided information and return a JSON object with the following structure. **Important: Your response must be enclosed within `<answer></answer>` tags.**

<answer>
```json
{
  "status": "approved|refinement_needed|limitation|other_issue",
  "translation_quality_is_excellent": boolean,
  "concrete_issues": "comprehensive description of specific translation quality issues",
  "templating_issues": "comprehensive description of issues related to the template structure (if applicable)",
  "other_issues": "comprehensive description of other issues beyond translation and templating issues (if applicable)",
}
```
</answer>

<status_values>
- **"approved"**: Translation quality is excellent and ready for use
- **"refinement_needed"**: Translation has issues that can be fixed through better translation
- **"limitation"**: Translation issues exist due to structural problems requiring code changes
- **"other_issue"**: Another issues exist that don't stem from poor translation or templating issues
</status_values>

<feedback_requirements>
- Provide specific examples from the generated problems that highlight translation issues
- Explain the reason for poor translation
- Suggest specific translation improvements where applicable
- **For human feedback**: Only include human feedback issues that are still applicable to the current translation. Do not penalize for issues that have already been resolved in the current translation.
- **For unaddressed previous LLM feedback**: Include any previously mentioned LLM issues that still exist in the current translation

First, think hard step by step and provide your reasoning within <thinking> and </thinking> tags.
</feedback_requirements>
</task_instructions>

</quality_assessment_criteria>