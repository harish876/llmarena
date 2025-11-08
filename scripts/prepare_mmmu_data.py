"""
Script to download and prepare MMMU dataset for matharena framework.

This script downloads MMMU data from HuggingFace and converts it to matharena's
expected format (CSV files and problem text files with images).
"""

import ast
import base64
import csv
import os
from pathlib import Path

from datasets import load_dataset
from loguru import logger
from PIL import Image
import io


def parse_options_string(options_str):
    """Parse the options string which is a string representation of a list."""
    try:
        # Remove any leading/trailing whitespace and parse
        options_str = options_str.strip()
        if options_str.startswith("'") and options_str.endswith("'"):
            options_str = options_str[1:-1]
        # Parse as Python literal
        options_list = ast.literal_eval(options_str)
        if isinstance(options_list, list):
            return options_list
        else:
            return [str(options_list)]
    except Exception as e:
        logger.warning(f"Failed to parse options string '{options_str}': {e}")
        # Fallback: try to split by comma
        return [opt.strip().strip("'\"") for opt in options_str.split(",")]


def format_multichoice_question(question: str, options_list: list, image_embeddings: list = None) -> str:
    """Format a multiple choice question with choices A, B, C, D, etc.
    
    Args:
        question: The question text
        options_list: List of answer options
        image_embeddings: List of base64-encoded image data URLs to embed in the text
    """
    formatted = f"{question}\n\n"
    
    # Embed images as base64 data URLs in the text
    # This allows models that support base64 images in text to process them directly
    if image_embeddings:
        formatted += "[IMAGES]\n"
        for i, img_data_url in enumerate(image_embeddings, 1):
            if len(image_embeddings) > 1:
                formatted += f"Image{i}:{img_data_url}\n"
            else:
                formatted += f"{img_data_url}\n"
        formatted += "[/IMAGES]\n\n"
    
    for i, option in enumerate(options_list):
        letter = chr(65 + i)  # A, B, C, D, etc.
        formatted += f"({letter}) {option}\n"
    return formatted


def get_all_images(examples):
    """Extract all non-null images from image_1 through image_7.
    
    Returns:
        List of PIL Image objects
    """
    images = []
    for i in range(1, 8):
        img_key = f"image_{i}"
        if img_key in examples and examples[img_key] is not None:
            images.append(examples[img_key])
    return images


def image_to_base64_data_url(image):
    """Convert a PIL Image to base64 data URL string.
    
    Args:
        image: PIL Image object
        
    Returns:
        Base64 data URL string (data:image/png;base64,...)
    """
    try:
        # Convert to bytes
        img_buffer = io.BytesIO()
        # Convert to RGB if necessary (some images might be RGBA)
        if image.mode in ('RGBA', 'LA', 'P'):
            # Convert RGBA to RGB for better compatibility
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            rgb_image.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
            image = rgb_image
        image.save(img_buffer, format='PNG')
        img_bytes = img_buffer.getvalue()
        
        # Encode to base64
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # Return as data URL
        return f"data:image/png;base64,{img_b64}"
    except Exception as e:
        logger.warning(f"Failed to convert image to base64: {e}")
        return None


def load_image_from_example(img_data):
    """Load PIL Image from various possible formats in the dataset.
    
    Args:
        img_data: Image data (could be PIL Image, dict with bytes, or bytes)
        
    Returns:
        PIL Image object or None
    """
    try:
        if hasattr(img_data, "save"):
            # Already a PIL Image
            return img_data
        elif isinstance(img_data, dict):
            # Image from datasets library
            if "bytes" in img_data:
                return Image.open(io.BytesIO(img_data["bytes"]))
            elif "path" in img_data:
                return Image.open(img_data["path"])
        elif isinstance(img_data, bytes):
            return Image.open(io.BytesIO(img_data))
        else:
            # Try to open as file path
            return Image.open(img_data)
    except Exception as e:
        logger.warning(f"Failed to load image: {e}")
        return None


def combine_images_vertically(images):
    """Combine multiple images vertically into a single image.
    
    Args:
        images: List of PIL Image objects
        
    Returns:
        Combined PIL Image
    """
    if not images:
        return None
    if len(images) == 1:
        return images[0]
    
    # Get maximum width
    max_width = max(img.width for img in images)
    
    # Resize all images to have the same width while maintaining aspect ratio
    resized_images = []
    total_height = 0
    for img in images:
        aspect_ratio = img.height / img.width
        new_height = int(max_width * aspect_ratio)
        resized_img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
        resized_images.append(resized_img)
        total_height += new_height
    
    # Create combined image
    combined = Image.new('RGB', (max_width, total_height), (255, 255, 255))
    y_offset = 0
    for img in resized_images:
        combined.paste(img, (0, y_offset))
        y_offset += img.height
    
    return combined


def prepare_mmmu_data(
    dataset_name: str = "MMMU/MMMU",
    config: str = "Clinical_Medicine",
    split: str = "validation",
    output_dir: str = "data/mmmu_clinical_medicine_validation",
):
    """
    Download and prepare MMMU data for matharena.

    Args:
        dataset_name: HuggingFace dataset name
        config: Dataset config name
        split: Dataset split (train/validation/test)
        output_dir: Output directory for matharena data
    """
    logger.info(f"Downloading MMMU dataset: {dataset_name}, config: {config}, split: {split}")
    try:
        dataset = load_dataset(dataset_name, config, split=split)
    except Exception as e:
        logger.error(f"Failed to load MMMU dataset: {e}")
        raise

    logger.info(f"Loaded {len(dataset)} examples from MMMU {config} {split}")

    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    problems_dir = output_path / "problems"
    problems_dir.mkdir(exist_ok=True)

    # Prepare answers.csv
    answers_data = []

    logger.info("Converting MMMU format to matharena format...")
    for idx, example in enumerate(dataset):
        problem_idx = idx + 1  # 1-indexed

        # Extract question and options
        question = example.get("question", "")
        options_str = example.get("options", "")
        answer = example.get("answer", "").strip()

        # Parse options
        options_list = parse_options_string(options_str)

        # Get all images
        image_data_list = get_all_images(example)
        images = []
        image_embeddings = []
        
        for img_data in image_data_list:
            img = load_image_from_example(img_data)
            if img is not None:
                images.append(img)
                # Convert to base64 data URL for embedding in text
                img_data_url = image_to_base64_data_url(img)
                if img_data_url:
                    image_embeddings.append(img_data_url)

        # Replace image placeholders in question with descriptive text
        # The question may contain <image 1>, <image 2>, etc.
        question_with_images = question
        for i in range(1, 8):
            placeholder = f"<image {i}>"
            if placeholder in question:
                question_with_images = question_with_images.replace(
                    placeholder, f"[Image {i} embedded below]"
                )

        # Format the problem with choices and embedded images
        problem_text = format_multichoice_question(question_with_images, options_list, image_embeddings)

        # Save problem text to file
        problem_file = problems_dir / f"{problem_idx}.tex"
        with open(problem_file, "w", encoding="utf-8") as f:
            f.write(problem_text)

        # Save images - combine multiple images into one for runner compatibility
        # Also save individual images if there are multiple
        if images:
            if len(images) == 1:
                # Single image - save as {id}.png
                image_file = problems_dir / f"{problem_idx}.png"
                images[0].save(image_file, "PNG")
            else:
                # Multiple images - combine them vertically for the main image file
                combined_image = combine_images_vertically(images)
                if combined_image:
                    image_file = problems_dir / f"{problem_idx}.png"
                    combined_image.save(image_file, "PNG")
                    logger.debug(f"Combined {len(images)} images for problem {problem_idx}")
                
                # Also save individual images for reference
                for i, img in enumerate(images, 1):
                    individual_image_file = problems_dir / f"{problem_idx}_image_{i}.png"
                    img.save(individual_image_file, "PNG")

        # Add to answers data
        answers_data.append({"id": problem_idx, "answer": answer})

        if (idx + 1) % 50 == 0:
            logger.info(f"Processed {idx + 1}/{len(dataset)} problems")

    # Write answers.csv
    answers_file = output_path / "answers.csv"
    with open(answers_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "answer"])
        writer.writeheader()
        writer.writerows(answers_data)

    logger.info(f"Successfully prepared MMMU data:")
    logger.info(f"  - {len(answers_data)} problems")
    logger.info(f"  - Problems saved to: {problems_dir}")
    logger.info(f"  - Answers saved to: {answers_file}")
    logger.info(f"\nYou can now run evaluations using:")
    logger.info(f"  python scripts/run_parallel.py --comp mmmu/mmmu_clinical_medicine --models <model> --n 1")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare MMMU data for matharena")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="MMMU/MMMU",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="Clinical_Medicine",
        help="Dataset config name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split (train/validation/test)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/mmmu_clinical_medicine_validation",
        help="Output directory",
    )
    args = parser.parse_args()

    prepare_mmmu_data(
        dataset_name=args.dataset_name,
        config=args.config,
        split=args.split,
        output_dir=args.output_dir,
    )

