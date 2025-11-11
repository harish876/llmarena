"""
Script to download and prepare AGIEval dataset for matharena framework.

This script downloads AGIEval data from the GitHub repository and converts
it to matharena's expected format (CSV files and problem text files).
"""

import csv
import json
import re
from pathlib import Path
from urllib.request import urlopen

from loguru import logger


def format_multichoice_question(passage: str, question: str, options: list) -> str:
    """
    Format a multiple choice question with passage, question, and options.
    
    Args:
        passage: Context/passage text (can be None or empty)
        question: The question text
        options: List of option strings, e.g., ["(A) option1", "(B) option2", ...]
    
    Returns:
        Formatted problem text
    """
    # Combine passage and question
    if passage and passage.strip():
        problem_text = f"{passage.strip()}\n\n{question.strip()}\n\n"
    else:
        problem_text = f"{question.strip()}\n\n"
    
    # Add options (they should already be formatted as "(A) ...", "(B) ...", etc.)
    for option in options:
        # Ensure option ends with newline
        option = option.strip()
        if option:
            problem_text += f"{option}\n"
    
    return problem_text


def parse_option_label(option: str) -> str:
    """
    Extract the label (A, B, C, etc.) from an option string.
    
    Args:
        option: Option string like "(A) option text" or "A) option text"
    
    Returns:
        The label letter (A, B, C, D, etc.)
    """
    # Match patterns like "(A)", "A)", "(A ", etc.
    match = re.match(r'^\(?([A-Z])\)?\s*', option.strip())
    if match:
        return match.group(1)
    return None


def download_jsonl(url: str) -> list:
    """
    Download and parse JSONL file from URL.
    
    Args:
        url: URL to the JSONL file
    
    Returns:
        List of JSON objects
    """
    logger.info(f"Downloading AGIEval data from {url}...")
    try:
        with urlopen(url) as response:
            data = response.read().decode('utf-8')
        lines = data.strip().split('\n')
        examples = [json.loads(line) for line in lines if line.strip()]
        logger.info(f"Loaded {len(examples)} examples from {url}")
        return examples
    except Exception as e:
        logger.error(f"Failed to download AGIEval data from {url}: {e}")
        raise


def prepare_agieval_data(
    dataset_name: str = "lsat-ar",
    output_dir: str = "data/agieval_lsat_ar",
    url: str = None,
    jsonl_file: str = None,
):
    """
    Download and prepare AGIEval data for matharena.
    
    Args:
        dataset_name: Name of the dataset (e.g., "lsat-ar")
        output_dir: Output directory for matharena data
        url: URL to the JSONL file (if None, will use default AGIEval GitHub URL)
        jsonl_file: Local path to JSONL file (if provided, will use this instead of downloading)
    """
    # Determine data source
    if jsonl_file:
        logger.info(f"Reading AGIEval data from local file: {jsonl_file}")
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            examples = [json.loads(line) for line in f if line.strip()]
        logger.info(f"Loaded {len(examples)} examples from {jsonl_file}")
    elif url:
        examples = download_jsonl(url)
    else:
        # Default to AGIEval GitHub repository
        default_url = f"https://raw.githubusercontent.com/ruixiangcui/AGIEval/refs/heads/main/data/v1_1/{dataset_name}.jsonl"
        examples = download_jsonl(default_url)
    
    # Validate required fields
    required_fields = ["question", "options", "label"]
    for idx, example in enumerate(examples):
        missing_fields = [field for field in required_fields if field not in example]
        if missing_fields:
            raise ValueError(f"Example {idx + 1} missing required fields: {missing_fields}")
    
    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    problems_dir = output_path / "problems"
    problems_dir.mkdir(exist_ok=True)
    
    # Prepare answers.csv
    answers_data = []
    
    logger.info("Converting AGIEval format to matharena format...")
    for idx, example in enumerate(examples):
        problem_idx = idx + 1  # 1-indexed
        
        # Extract fields
        passage = example.get("passage", "")
        question = example.get("question", "")
        options = example.get("options", [])
        label = example.get("label", "").strip().upper()
        
        # Validate label
        if not label or len(label) != 1 or label not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            logger.warning(f"Example {idx + 1} has invalid label: {label}. Skipping.")
            continue
        
        # Parse options to ensure they're in the correct format
        # Options should already be formatted as "(A) ...", but we'll verify
        formatted_options = []
        for option in options:
            option_str = str(option).strip()
            if option_str:
                # If option doesn't start with a letter label, try to infer from position
                if not re.match(r'^\(?[A-Z]\)?\s*', option_str):
                    # This shouldn't happen in AGIEval, but handle it gracefully
                    logger.warning(f"Example {idx + 1} option doesn't have label: {option_str}")
                formatted_options.append(option_str)
        
        # Format the problem
        problem_text = format_multichoice_question(passage, question, formatted_options)
        
        # Save problem text to file
        problem_file = problems_dir / f"{problem_idx}.tex"
        with open(problem_file, "w", encoding="utf-8") as f:
            f.write(problem_text)
        
        # Add to answers data
        answers_data.append({"id": problem_idx, "answer": label})
        
        if (idx + 1) % 100 == 0:
            logger.info(f"Processed {idx + 1}/{len(examples)} problems")
    
    # Write answers.csv
    answers_file = output_path / "answers.csv"
    with open(answers_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "answer"])
        writer.writeheader()
        writer.writerows(answers_data)
    
    logger.info(f"Successfully prepared AGIEval data:")
    logger.info(f"  - {len(answers_data)} problems")
    logger.info(f"  - Problems saved to: {problems_dir}")
    logger.info(f"  - Answers saved to: {answers_file}")
    logger.info(f"\nYou can now run evaluations using:")
    logger.info(f"  python scripts/run_parallel.py --comp agieval/{dataset_name.replace('-', '_')} --models <model> --n 4")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare AGIEval data for matharena")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="lsat-ar",
        help="AGIEval dataset name (e.g., 'lsat-ar', 'lsat-lr', 'lsat-rc')"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/agieval_<dataset_name>)"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="URL to JSONL file (overrides default GitHub URL)"
    )
    parser.add_argument(
        "--jsonl-file",
        type=str,
        default=None,
        help="Local path to JSONL file (overrides URL)"
    )
    args = parser.parse_args()
    
    # Set default output dir if not provided
    if args.output_dir is None:
        # Convert dataset name to valid directory name
        dir_name = args.dataset_name.replace("-", "_")
        args.output_dir = f"data/agieval_{dir_name}"
    
    prepare_agieval_data(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        url=args.url,
        jsonl_file=args.jsonl_file,
    )

