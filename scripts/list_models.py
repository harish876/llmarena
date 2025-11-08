#!/usr/bin/env python3
"""Script to list all models used for a given competition."""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

from loguru import logger


def list_models_for_competition(competition_name: str, output_dir: str = "outputs"):
    """
    Lists all models used for a given competition.
    
    Args:
        competition_name: Name of the competition (e.g., "gpqa/gpqa_diamond" or "aime/aime_2025")
        output_dir: Directory containing competition outputs (default: "outputs")
    
    Returns:
        dict: Dictionary mapping API providers to lists of model names
    """
    competition_path = Path(output_dir) / competition_name
    
    if not competition_path.exists():
        logger.error(f"Competition directory not found: {competition_path}")
        return {}
    
    models_by_api = defaultdict(list)
    
    # Iterate through API directories (bedrock, openrouter, etc.)
    for api_dir in competition_path.iterdir():
        if not api_dir.is_dir():
            continue
        
        api_name = api_dir.name
        # Skip non-API directories like leaderboard files
        if api_name.endswith(".json") or api_name.endswith(".jsonl"):
            continue
        
        # Iterate through model directories
        for model_dir in api_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_name = model_dir.name
            # Check if this directory contains actual run files (JSON files)
            json_files = list(model_dir.glob("*.json"))
            if json_files:
                models_by_api[api_name].append(model_name)
    
    return dict(models_by_api)


def get_flattened_model_list(models_by_api: dict) -> list:
    """
    Flattens the models_by_api dictionary into a list of 'api/model' strings.
    
    Args:
        models_by_api: Dictionary mapping API providers to lists of model names
    
    Returns:
        list: List of model paths in format 'api/model'
    """
    all_models = []
    for api_name in sorted(models_by_api.keys()):
        for model in sorted(models_by_api[api_name]):
            all_models.append(f"{api_name}/{model}")
    return all_models


def print_models(models_by_api: dict, competition_name: str):
    """Prints the models in a formatted way."""
    if not models_by_api:
        print(f"\nNo models found for competition: {competition_name}\n")
        return
    
    print(f"\n{'='*60}")
    print(f"Models for competition: {competition_name}")
    print(f"{'='*60}\n")
    
    total_models = 0
    for api_name in sorted(models_by_api.keys()):
        models = sorted(models_by_api[api_name])
        total_models += len(models)
        print(f"  {api_name.upper()}:")
        for model in models:
            print(f"    - {model}")
        print()
    
    print(f"{'='*60}")
    print(f"Total: {total_models} models across {len(models_by_api)} API provider(s)")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="List all models used for a given competition"
    )
    parser.add_argument(
        "--comp",
        type=str,
        required=True,
        help="Competition name (e.g., 'gpqa/gpqa_diamond' or 'aime/aime_2025')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory containing competition outputs (default: outputs)",
    )
    parser.add_argument(
        "--api",
        type=str,
        default=None,
        help="Filter by specific API provider (e.g., 'bedrock', 'openrouter')",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["pretty", "list", "csv"],
        default="pretty",
        help="Output format (default: pretty)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to output JSON file with model list (default: model_list.json in competition directory, or current directory if competition dir doesn't exist)",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Do not generate JSON file",
    )
    
    args = parser.parse_args()
    
    models_by_api = list_models_for_competition(args.comp, args.output_dir)
    
    if args.api:
        # Filter by API
        if args.api in models_by_api:
            models_by_api = {args.api: models_by_api[args.api]}
        else:
            logger.warning(f"API '{args.api}' not found for competition '{args.comp}'")
            models_by_api = {}
    
    # Generate flattened model list for JSON output
    flattened_models = get_flattened_model_list(models_by_api)
    
    # Generate JSON file if not disabled
    if not args.no_json and flattened_models:
        if args.output_json:
            json_path = Path(args.output_json)
        else:
            # Default: save in competition directory, or current directory if competition dir doesn't exist
            competition_path = Path(args.output_dir) / args.comp
            if competition_path.exists():
                json_path = competition_path / "model_list.json"
            else:
                json_path = Path("model_list.json")
        
        # Ensure parent directory exists
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write JSON file
        with open(json_path, "w") as f:
            json.dump(flattened_models, f, indent=2)
        
        logger.info(f"Model list saved to: {json_path}")
    
    # Display to console based on format
    if args.format == "pretty":
        print_models(models_by_api, args.comp)
    elif args.format == "list":
        # Simple list format (one model per line)
        # If API filter is specified, only show model names; otherwise show full paths
        if args.api:
            for api_name in sorted(models_by_api.keys()):
                for model in sorted(models_by_api[api_name]):
                    print(model)
        else:
            for model in flattened_models:
                print(model)
    elif args.format == "csv":
        # CSV format: api,model
        print("api,model")
        for model_path in flattened_models:
            api_name, model_name = model_path.split("/", 1)
            print(f"{api_name},{model_name}")


if __name__ == "__main__":
    main()

