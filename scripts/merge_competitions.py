#!/usr/bin/env python3
"""Script to merge results from two competitions by model name."""

import argparse
import os
import shutil
from pathlib import Path
from loguru import logger

def merge_competitions(comp1_dir, comp2_dir, output_dir, comp1_name, comp2_name):
    """
    Merge results from two competitions by model name.
    
    Args:
        comp1_dir: Path to first competition directory
        comp2_dir: Path to second competition directory
        output_dir: Path to output merged directory
        comp1_name: Name of first competition (for logging)
        comp2_name: Name of second competition (for logging)
    """
    comp1_path = Path(comp1_dir)
    comp2_path = Path(comp2_dir)
    output_path = Path(output_dir)
    
    if not comp1_path.exists():
        raise ValueError(f"Competition directory does not exist: {comp1_dir}")
    if not comp2_path.exists():
        raise ValueError(f"Competition directory does not exist: {comp2_dir}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all provider directories (e.g., openrouter, bedrock)
    providers = set()
    if comp1_path.exists():
        providers.update([d.name for d in comp1_path.iterdir() if d.is_dir()])
    if comp2_path.exists():
        providers.update([d.name for d in comp2_path.iterdir() if d.is_dir()])
    
    logger.info(f"Found providers: {providers}")
    
    total_files = 0
    for provider in providers:
        provider_comp1 = comp1_path / provider
        provider_comp2 = comp2_path / provider
        provider_output = output_path / provider
        
        if not provider_comp1.exists() and not provider_comp2.exists():
            continue
        
        provider_output.mkdir(parents=True, exist_ok=True)
        
        # Get all models from both competitions
        models = set()
        if provider_comp1.exists():
            models.update([d.name for d in provider_comp1.iterdir() if d.is_dir()])
        if provider_comp2.exists():
            models.update([d.name for d in provider_comp2.iterdir() if d.is_dir()])
        
        logger.info(f"Found {len(models)} models in provider {provider}")
        
        for model in models:
            model_comp1 = provider_comp1 / model
            model_comp2 = provider_comp2 / model
            model_output = provider_output / model
            
            model_output.mkdir(parents=True, exist_ok=True)
            
            # Copy files from comp1
            files_copied = 0
            if model_comp1.exists():
                for file in model_comp1.glob("*.json"):
                    dest = model_output / file.name
                    if dest.exists():
                        logger.warning(f"File already exists, skipping: {dest}")
                        continue
                    shutil.copy2(file, dest)
                    files_copied += 1
                    total_files += 1
                logger.info(f"Copied {files_copied} files from {comp1_name}/{provider}/{model}")
            
            # Copy files from comp2
            files_copied = 0
            if model_comp2.exists():
                for file in model_comp2.glob("*.json"):
                    dest = model_output / file.name
                    if dest.exists():
                        logger.warning(f"File already exists, skipping: {dest}")
                        continue
                    shutil.copy2(file, dest)
                    files_copied += 1
                    total_files += 1
                logger.info(f"Copied {files_copied} files from {comp2_name}/{provider}/{model}")
    
    logger.info(f"Merge complete! Total files copied: {total_files}")
    logger.info(f"Merged results saved to: {output_path}")
    
    # Copy leaderboard files if they exist
    for comp_path, comp_name in [(comp1_path, comp1_name), (comp2_path, comp2_name)]:
        leaderboard_file = comp_path / "leaderboard_summary.json"
        if leaderboard_file.exists():
            logger.info(f"Found leaderboard file in {comp_name}, but not copying (merged leaderboard should be regenerated)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge results from two competitions by model name")
    parser.add_argument("--comp1", type=str, required=True, help="Path to first competition directory")
    parser.add_argument("--comp2", type=str, required=True, help="Path to second competition directory")
    parser.add_argument("--output", type=str, required=True, help="Path to output merged directory")
    parser.add_argument("--comp1-name", type=str, help="Name of first competition (for logging)")
    parser.add_argument("--comp2-name", type=str, help="Name of second competition (for logging)")
    
    args = parser.parse_args()
    
    comp1_name = args.comp1_name or os.path.basename(os.path.normpath(args.comp1))
    comp2_name = args.comp2_name or os.path.basename(os.path.normpath(args.comp2))
    
    merge_competitions(
        args.comp1,
        args.comp2,
        args.output,
        comp1_name,
        comp2_name
    )


