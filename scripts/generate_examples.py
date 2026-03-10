#!/usr/bin/env python3
"""
Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""

"""
Generate example outputs for Reasoning Gym tasks.
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import reasoning_gym
from reasoning_gym.multilingual import TranslationManager


LANG_NAMES = {
    "en_US": "English",
    "es_ES": "Spanish",
    "fr_FR": "French",
    "de_DE": "German",
    "it_IT": "Italian",
    "pt_BR": "Portuguese",
    "ru_RU": "Russian",
    "ja_JP": "Japanese",
    "ko_KR": "Korean",
    "zh_CN": "Chinese",
    "th_TH": "Thai",
    "sw_KE": "Swahili",
    "te_IN": "Telugu",
    "bn_BD": "Bengali",
}


@dataclass
class Example:
    """Represents a single task example in a specific locale."""

    task_name: str
    locale: str
    question: str
    answer: str


def generate_examples(
    tasks: List[str],
    locales: List[str],
    num_examples: int = 5,
    seed: int = 42,
) -> List[Example]:
    """Generate examples for specified tasks and locales.

    Args:
        tasks: List of task keys (e.g., ["basic_arithmetic"]).
        locales: List of locale codes (e.g., ["en_US", "es_ES"]).
        num_examples: Number of examples to generate per task-locale combination.
        seed: Random seed for reproducibility.

    Returns:
        List of Example objects containing questions and answers across tasks and locales.
    """
    tm = TranslationManager()
    tasks_by_group = tm.get_available_tasks()

    task_paths = {}
    for group, group_tasks in tasks_by_group.items():
        for task in group_tasks:
            if task in tasks:
                task_paths[task] = f"{group}/{task}"

    examples = []
    for task_key, task_path in sorted(task_paths.items()):
        for locale in sorted(locales):
            lang = locale.split("_")[0]

            dataset = reasoning_gym.create_dataset(
                task_key, size=num_examples, seed=seed, languages=lang
            )

            for item in dataset:
                examples.append(
                    Example(
                        task_name=task_path,
                        locale=locale,
                        question=item["question"],
                        answer=item["answer"],
                    )
                )

    return examples


def format_as_jsonl(
    examples: List[Example],
    output_dir: Path,
) -> None:
    """Format examples as JSONL files, one per non-English locale.

    Creates paired English/target locale examples and saves them to [locale].jsonl files.

    Args:
        examples: List of Example objects (should include both English and target locales).
        output_dir: Directory where JSONL files will be written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    by_locale_task: Dict[str, Dict[str, List[Example]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for ex in examples:
        by_locale_task[ex.locale][ex.task_name].append(ex)

    for locale, tasks_dict in sorted(by_locale_task.items()):
        if locale.startswith("en"):
            continue

        jsonl_lines = []

        for task_name, target_examples in sorted(tasks_dict.items()):
            eng_examples = by_locale_task["en_US"][task_name]

            for i, target_example in enumerate(target_examples):
                jsonl_entry = {
                    "locale_id": locale,
                    "target_language": LANG_NAMES[locale],
                    "task_name": task_name,
                    "english_question": eng_examples[i].question,
                    "english_answer": eng_examples[i].answer,
                    "target_question": target_example.question,
                    "target_answer": target_example.answer,
                }
                jsonl_lines.append(json.dumps(jsonl_entry, ensure_ascii=False))

        output_file = output_dir / f"{locale}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(jsonl_lines))
        print(f"Written {len(jsonl_lines)} examples to {output_file}")


def format_as_markdown(
    examples: List[Example],
    output_dir: Path,
) -> None:
    """Format examples as a single Markdown file with parallel examples across all locales.

    Shows all language versions of each example side by side.

    Args:
        examples: List of Example objects.
        output_dir: Directory where Markdown file will be written.
        lang_names: Mapping of locale codes to language names.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    by_task: Dict[str, Dict[str, List[Example]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for ex in examples:
        by_task[ex.task_name][ex.locale].append(ex)

    all_locales = set()
    for task_dict in by_task.values():
        all_locales.update(task_dict.keys())

    lines = []
    lines.append("# Examples\n\n")

    for task_name, locale_dict in sorted(by_task.items()):
        lines.append(f"## {task_name}\n\n")
        max_examples = max(len(exs) for exs in locale_dict.values())

        for i in range(max_examples):
            lines.append(f"### Example {i + 1}\n\n")

            for locale in sorted(all_locales):
                example = locale_dict[locale][i]
                language_name = LANG_NAMES[locale]

                lines.append(
                    f"#### {language_name}\n\n"
                    f"**Question:**\n"
                    f"`````\n"
                    f"{example.question}\n"
                    f"`````\n\n"
                    f"**Answer:**\n"
                    f"`````\n"
                    f"{example.answer}\n"
                    f"`````\n\n"
                )

            if i < max_examples - 1:
                lines.append("-------------\n\n")

        lines.append("\n")

    output_file = output_dir / "examples.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    print(f"Written {len(by_task)} tasks to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate example outputs for Reasoning Gym tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate examples for all tasks in Spanish and French
  %(prog)s --filter-locales es_ES,fr_FR

  # Generate examples for specific tasks in all available locales
  %(prog)s --filter-tasks basic_arithmetic,word_sorting

  # Generate 10 examples with custom seed
  %(prog)s --num-examples 10 --seed 123

  # Generate for specific tasks and locales
  %(prog)s --filter-tasks basic_arithmetic --filter-locales es_ES,zh_CN,ja_JP --output-dir my_examples
        """,
    )

    parser.add_argument(
        "--filter-tasks",
        type=str,
        help="Comma-separated list of task keys to generate (e.g., 'basic_arithmetic,word_sorting'). "
        "If not specified, generates for all available tasks.",
    )

    parser.add_argument(
        "--filter-locales",
        type=str,
        help="Comma-separated list of locale codes to generate (e.g., 'es_ES,fr_FR,zh_CN'). "
        "If not specified, generates for all available locales.",
    )

    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="Number of examples to generate per task-language combination (default: 5)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="examples",
        help="Output directory for generated files (default: examples)",
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["jsonl", "markdown", "both"],
        default="both",
        help="Output format (default: both)",
    )

    args = parser.parse_args()

    tm = TranslationManager()
    tasks_by_group = tm.get_available_tasks()
    all_tasks = []
    for _, group_tasks in tasks_by_group.items():
        all_tasks.extend(group_tasks)

    if args.filter_tasks:
        tasks = [t.strip() for t in args.filter_tasks.split(",")]
    else:
        tasks = sorted(all_tasks)

    if args.filter_locales:
        locales = [l.strip() for l in args.filter_locales.split(",")]
    else:
        locales = sorted(list(LANG_NAMES.keys()))

    print(f"Generating examples with seed={args.seed}...")
    print(f"  Tasks: {', '.join(tasks) if len(tasks) < 5 else f'{len(tasks)} tasks'}")
    print(f"  Locales: {', '.join(locales)}")

    examples = generate_examples(
        tasks=tasks,
        locales=locales,
        num_examples=args.num_examples,
        seed=args.seed,
    )
    print(f"\nGenerated {len(examples)} total examples")

    output_dir = Path(args.output_dir)

    if args.format in ["jsonl", "both"]:
        print("\nGenerating JSONL files...")
        format_as_jsonl(examples, output_dir)

    if args.format in ["markdown", "both"]:
        print("\nGenerating Markdown files...")
        format_as_markdown(examples, output_dir)

    print(f"\nAll files written to {output_dir.absolute()}")

if __name__ == "__main__":
    main()
