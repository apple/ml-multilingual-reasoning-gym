#!/usr/bin/env python3
"""
Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""
"""
Script to generate Reasoning Gym datasets and save them to the Hugging Face Hub.
"""

import argparse
from typing import Dict, List, Optional

import yaml
from datasets import Dataset
from tqdm import tqdm

from reasoning_gym.composite import DatasetSpec
from reasoning_gym.factory import DATASETS, create_dataset
from reasoning_gym.utils import maybe_decompose_dataset_name


def generate_dataset(
    dataset_names: List[str],
    dataset_size: int = 20000,
    seed: int = 42,
    weights: Optional[Dict[str, float]] = None,
    configs: Optional[Dict[str, Dict]] = None,
) -> Dataset:
    """
    Generate a dataset from the specified Reasoning Gym datasets.

    Args:
        dataset_names: List of dataset names to include
        dataset_size: Total size of the dataset to generate
        seed: Random seed for dataset generation
        weights: Optional dictionary mapping dataset names to weights
        configs: Optional dictionary mapping dataset names to configurations

    Returns:
        A Hugging Face Dataset object
    """
    # Validate dataset names (decompose language suffixes first)
    for name in dataset_names:
        base_name, _ = maybe_decompose_dataset_name(name)
        if base_name not in DATASETS:
            raise ValueError(f"Dataset '{base_name}' not found. Available datasets: {sorted(DATASETS.keys())}")

    # Set default weights if not provided
    if weights is None:
        equal_weight = 1.0 / len(dataset_names)
        weights = {name: equal_weight for name in dataset_names}
    else:
        # Validate weights
        for name in dataset_names:
            if name not in weights:
                weights[name] = 0.0
                print(f"Warning: No weight provided for {name}, setting to 0.0")

    # Set default configs if not provided
    if configs is None:
        configs = {name: {} for name in dataset_names}
    else:
        # Add empty configs for missing datasets
        for name in dataset_names:
            if name not in configs:
                configs[name] = {}

    # Create dataset specs
    dataset_specs = [DatasetSpec(name=name, weight=weights[name], config=configs[name]) for name in dataset_names]

    # Create composite dataset
    data_source = create_dataset("composite", seed=seed, size=dataset_size, datasets=dataset_specs)

    # Generate all examples
    examples = []
    for idx in tqdm(range(dataset_size), desc="Generating examples"):
        example = data_source[idx]
        examples.append(example)

    # Convert to HF Dataset
    hf_dataset = Dataset.from_list(examples)
    return hf_dataset


def save_to_hub(
    dataset: Dataset,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False,
    commit_message: str = "Upload reasoning_gym dataset",
    split: Optional[str] = None,
) -> str:
    """
    Save the dataset to the Hugging Face Hub.

    Args:
        dataset: HF Dataset to save
        repo_id: Hugging Face repo ID (e.g., "username/dataset-name")
        token: HF API token
        private: Whether the repository should be private
        commit_message: Commit message
        split: Dataset split name

    Returns:
        URL of the uploaded dataset
    """
    # Push to the hub
    dataset.push_to_hub(
        repo_id,
        token=token,
        private=private,
        commit_message=commit_message,
    )

    print(f"Dataset pushed to https://huggingface.co/datasets/{repo_id}")
    return f"https://huggingface.co/datasets/{repo_id}"


def load_config(config_path: str) -> dict:
    """
    Load dataset configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing the configuration
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Generate and upload Reasoning Gym datasets to HF Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Multiple datasets with different language combinations
  %(prog)s --dataset "basic_arithmetic:en,de;spell_backward:fr,es;word_sorting:zh,ja" --size 5000 --repo-id "username/dataset-name"

  # Using a config file
  %(prog)s --config example_hf_dataset_config.yaml
        """,
    )
    parser.add_argument("--dataset", type=str, required=False, help="Dataset names (semicolon-separated list, e.g., 'dataset1:en,de;dataset2:fr,es')")
    parser.add_argument("--config", type=str, required=False, help="Path to dataset configuration YAML file")
    parser.add_argument("--size", type=int, default=None, help="Total dataset size (default: 20000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--repo-id", type=str, help="Hugging Face repository ID (e.g., 'username/dataset-name')")
    parser.add_argument("--private", action="store_true", help="Make the HF repository private")
    parser.add_argument(
        "--split", type=str, choices=["train", "test", "validation"], default=None, help="Dataset split name (default: train)"
    )

    args = parser.parse_args()

    # Load configuration
    dataset_names = []
    weights = {}
    configs = {}
    repo_id = args.repo_id
    private = args.private
    split = args.split if args.split is not None else "train"
    size = args.size if args.size is not None else 20000
    seed = args.seed

    # Load from config file if provided
    if args.config:
        config = load_config(args.config)
        if "reasoning_gym" in config:
            rg_config = config["reasoning_gym"]
            if "datasets" in rg_config:
                for name, ds_config in rg_config["datasets"].items():
                    dataset_names.append(name)
                    weights[name] = ds_config.get("weight", 1.0 / len(rg_config["datasets"]))
                    configs[name] = ds_config.get("config", {})

            # Use config size only if not provided via CLI
            if "dataset_size" in rg_config and args.size is None:
                size = rg_config["dataset_size"]

        # Use config HF settings only if not provided via CLI
        if "huggingface" in config:
            hf_config = config["huggingface"]
            if not repo_id and "repo_id" in hf_config:
                repo_id = hf_config["repo_id"]
            if not args.private and "private" in hf_config:
                private = hf_config["private"]
            if args.split is None and "split" in hf_config:
                split = hf_config["split"]

    # Override datasets if provided via CLI
    if args.dataset:
        dataset_names = [name.strip() for name in args.dataset.split(";")]
        # Reset weights and configs when overriding via CLI
        equal_weight = 1.0 / len(dataset_names)
        weights = {name: equal_weight for name in dataset_names}
        configs = {}

    # Validate inputs
    if not repo_id:
        parser.error("--repo-id is required. Provide it via command line or in config file under huggingface.repo_id")

    if not dataset_names:
        parser.error("--dataset is required. Provide it via command line or in config file under reasoning_gym.datasets")

    print(f"Generating dataset with {len(dataset_names)} datasets: {', '.join(dataset_names)}")
    print(f"Dataset size: {size}")
    print(f"Dataset seed: {seed}")
    print(f"Repository ID: {repo_id}")

    # Generate the dataset
    dataset = generate_dataset(
        dataset_names=dataset_names,
        dataset_size=size,
        seed=seed,
        weights=weights,
        configs=configs,
    )

    # Save to hub with specified split
    save_to_hub(
        dataset=dataset,
        repo_id=repo_id,
        private=private,
        commit_message=f"Upload reasoning_gym dataset with {len(dataset_names)} datasets: {', '.join(dataset_names)}",
        split=split,
    )

    print("Done!")


if __name__ == "__main__":
    main()
