"""
Script to download and prepare GPQA dataset for matharena framework.

This script downloads GPQA data from the OpenAI public blob storage and converts
it to matharena's expected format (CSV files and problem text files).
"""

import csv
import os
import random
from pathlib import Path

import pandas as pd
from loguru import logger


def format_multichoice_question(question: str, choices: dict) -> str:
    """Format a multiple choice question with choices A, B, C, D."""
    formatted = f"{question}\n\n"
    for letter in ["A", "B", "C", "D"]:
        if letter in choices:
            formatted += f"({letter}) {choices[letter]}\n"
    return formatted


def prepare_gpqa_data(variant: str = "diamond", output_dir: str = "data/gpqa_diamond", seed: int = 0):
    """
    Download and prepare GPQA data for matharena.

    Args:
        variant: GPQA variant (e.g., "diamond")
        output_dir: Output directory for matharena data
        seed: Random seed for permutation reproducibility
    """
    logger.info(f"Downloading GPQA {variant} dataset...")
    url = f"https://openaipublic.blob.core.windows.net/simple-evals/gpqa_{variant}.csv"
    try:
        df = pd.read_csv(url)
    except Exception as e:
        logger.error(f"Failed to download GPQA data from {url}: {e}")
        raise

    logger.info(f"Loaded {len(df)} examples from GPQA {variant}")

    # Validate required columns
    required_columns = ["Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in GPQA data: {missing_columns}")

    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    problems_dir = output_path / "problems"
    problems_dir.mkdir(exist_ok=True)

    # Prepare answers.csv
    answers_data = []
    rng = random.Random(seed)

    logger.info("Converting GPQA format to matharena format...")
    for idx, row in df.iterrows():
        problem_idx = idx + 1  # 1-indexed

        # Get choices
        choices_list = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]

        # Permute choices (same as GPQA eval does)
        permutation = rng.sample(range(4), 4)
        permuted_choices = [choices_list[i] for i in permutation]

        # Find correct answer letter after permutation
        correct_index = permuted_choices.index(row["Correct Answer"])
        correct_answer = "ABCD"[correct_index]

        # Format the problem with choices
        choices_dict = {
            "A": permuted_choices[0],
            "B": permuted_choices[1],
            "C": permuted_choices[2],
            "D": permuted_choices[3],
        }
        problem_text = format_multichoice_question(row["Question"], choices_dict)

        # Save problem text to file
        problem_file = problems_dir / f"{problem_idx}.tex"
        with open(problem_file, "w", encoding="utf-8") as f:
            f.write(problem_text)

        # Add to answers data
        answers_data.append({"id": problem_idx, "answer": correct_answer})

        if (idx + 1) % 100 == 0:
            logger.info(f"Processed {idx + 1}/{len(df)} problems")

    # Write answers.csv
    answers_file = output_path / "answers.csv"
    with open(answers_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "answer"])
        writer.writeheader()
        writer.writerows(answers_data)

    logger.info(f"Successfully prepared GPQA data:")
    logger.info(f"  - {len(answers_data)} problems")
    logger.info(f"  - Problems saved to: {problems_dir}")
    logger.info(f"  - Answers saved to: {answers_file}")
    logger.info(f"\nYou can now run evaluations using:")
    logger.info(f"  python scripts/run_parallel.py --comp gpqa/gpqa_diamond --models <model> --n 4")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare GPQA data for matharena")
    parser.add_argument("--variant", type=str, default="diamond", help="GPQA variant (e.g., 'diamond')")
    parser.add_argument("--output-dir", type=str, default="data/gpqa_diamond", help="Output directory")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for permutation")
    args = parser.parse_args()

    prepare_gpqa_data(variant=args.variant, output_dir=args.output_dir, seed=args.seed)

