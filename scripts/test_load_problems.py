"""
Test script for loading problems and displaying sample data.

Usage:
    python scripts/test_load_problems.py mmmu/mmmu_clinical_medicine
    python scripts/test_load_problems.py configs/competitions/mmmu/mmmu_clinical_medicine.yaml
"""

import argparse
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from matharena.runner import Runner

def main(config_input):
    """Load problems and print first problem/answer and total count."""
    config_path = Path(config_input)
    
    # If the input doesn't exist as a file, try constructing the path
    if not config_path.exists():
        comp_configs_dir = Path("configs/competitions")
        config_path = comp_configs_dir / f"{config_input}.yaml"
        
        if not config_path.exists():
            print(f"Error: Config file not found: {config_input}")
            sys.exit(1)
    
    # Extract competition name from config path
    try:
        rel_path = config_path.relative_to(Path("configs/competitions"))
        comp_name = str(rel_path.with_suffix(""))
    except ValueError:
        comp_name = str(config_path.with_suffix(""))
    
    runner = Runner(
        comp_name=comp_name,
        runs_per_problem=1,
        comp_configs_dir="configs/competitions",
        solver_configs_dir="configs/models",
        output_dir="outputs",
        redo_all=False
    )
    
    problems = runner.problems
    
    print(f"Total rows: {len(problems)}")
    
    if problems:
        first_problem = problems[0]
        print(f"\nProblem: {first_problem['problem']}")
        print(f"Answer: {first_problem['answer']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load problems and display sample data"
    )
    parser.add_argument(
        "config_file",
        type=str,
        help="Competition name (e.g., mmmu/mmmu_clinical_medicine) or full path to config file"
    )
    
    args = parser.parse_args()
    main(args.config_file)
