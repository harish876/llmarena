import os
import json
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from loguru import logger
import argparse
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, desc=None):
        if desc:
            logger.info(desc)
        return iterable


def upload_file_to_s3(
    s3_client,
    local_path: str,
    bucket: str,
    s3_key: str,
    content_type: str = "application/json",
) -> bool:
    """Upload a single file to S3.
    
    Args:
        s3_client: Boto3 S3 client
        local_path: Local file path
        bucket: S3 bucket name
        s3_key: S3 object key (path within bucket)
        content_type: Content type for the file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(local_path, "rb") as f:
            s3_client.upload_fileobj(
                f,
                bucket,
                s3_key,
                ExtraArgs={"ContentType": content_type},
            )
        return True
    except Exception as e:
        logger.error(f"Failed to upload {local_path} to s3://{bucket}/{s3_key}: {e}")
        return False


def upload_outputs_to_s3(
    outputs_dir: str,
    bucket: str,
    s3_prefix: str = "outputs",
    competition: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    region: str = "us-east-1",
    endpoint_url: Optional[str] = None,
) -> None:
    """Upload outputs directory to S3.
    
    Args:
        outputs_dir: Local path to outputs directory
        bucket: S3 bucket name
        s3_prefix: Prefix for S3 keys (default: "outputs")
        competition: Optional competition name to filter (e.g., "aime")
        provider: Optional provider name to filter (e.g., "bedrock", "openrouter")
        model: Optional model name to filter
        region: AWS region
        endpoint_url: Optional S3 endpoint URL (for S3-compatible services)
    """
    # Initialize S3 client
    try:
        s3_client = boto3.client(
            "s3",
            region_name=region,
            endpoint_url=endpoint_url,
        )
        # Test credentials by checking bucket access
        s3_client.head_bucket(Bucket=bucket)
        logger.info(f"Successfully connected to S3 bucket: {bucket}")
    except NoCredentialsError:
        logger.error("AWS credentials not found. Please configure AWS credentials.")
        raise
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code == "404":
            logger.error(f"Bucket {bucket} does not exist.")
        elif error_code == "403":
            logger.error(f"Access denied to bucket {bucket}. Check permissions.")
        else:
            logger.error(f"Failed to access bucket {bucket}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize S3 client: {e}")
        raise

    outputs_path = Path(outputs_dir)
    if not outputs_path.exists():
        logger.error(f"Outputs directory {outputs_dir} does not exist.")
        raise ValueError(f"Directory {outputs_dir} not found")

    # Collect all files to upload
    files_to_upload = []

    # Build search pattern - handle nested paths like "aime/aime_2025"
    search_path = outputs_path
    if competition:
        # Split competition path by '/' and join with Path
        competition_parts = competition.split('/')
        search_path = outputs_path
        for part in competition_parts:
            search_path = search_path / part
        if not search_path.exists():
            logger.error(f"Competition directory {search_path} does not exist.")
            raise ValueError(f"Directory {search_path} not found")
        logger.info(f"Searching for files under {search_path}")

    # Find all JSON files recursively
    for json_file in search_path.rglob("*.json"):
        rel_path = json_file.relative_to(outputs_path)
        path_parts = rel_path.parts
        
        # Apply provider filter if specified
        # Structure: competition/[subdirs]/provider/model/file.json
        # We need to find provider in the path parts
        if provider:
            provider_found = False
            for part in path_parts:
                if part == provider:
                    provider_found = True
                    break
            if not provider_found:
                continue
        
        # Apply model filter if specified
        # Model typically comes after provider
        if model:
            model_found = False
            for i, part in enumerate(path_parts):
                if part == model:
                    model_found = True
                    break
            if not model_found:
                continue
        
        # Build S3 key preserving the full relative path
        s3_key = f"{s3_prefix}/{rel_path.as_posix()}"
        files_to_upload.append((json_file, s3_key))

    if not files_to_upload:
        logger.warning("No files found to upload.")
        return

    logger.info(f"Found {len(files_to_upload)} files to upload to s3://{bucket}/{s3_prefix}/")

    # Upload files with progress bar
    successful = 0
    failed = 0
    
    for local_file, s3_key in tqdm(files_to_upload, desc="Uploading to S3"):
        if upload_file_to_s3(s3_client, str(local_file), bucket, s3_key):
            successful += 1
        else:
            failed += 1

    logger.info(f"Upload complete: {successful} successful, {failed} failed")


def list_outputs_on_s3(
    bucket: str,
    s3_prefix: str = "outputs",
    competition: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    region: str = "us-east-1",
    endpoint_url: Optional[str] = None,
) -> list:
    """List output files in S3.
    
    Args:
        bucket: S3 bucket name
        s3_prefix: Prefix for S3 keys
        competition: Optional competition name to filter
        provider: Optional provider name to filter
        model: Optional model name to filter
        region: AWS region
        endpoint_url: Optional S3 endpoint URL
        
    Returns:
        List of S3 keys matching the filters
    """
    try:
        s3_client = boto3.client(
            "s3",
            region_name=region,
            endpoint_url=endpoint_url,
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
        )
    except NoCredentialsError:
        logger.error("AWS credentials not found. Please configure AWS credentials.")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize S3 client: {e}")
        raise

    prefix = f"{s3_prefix}/"
    if competition:
        prefix += f"{competition}/"
    if provider:
        prefix += f"{provider}/"
    if model:
        prefix += f"{model}/"

    results = []
    paginator = s3_client.get_paginator("list_objects_v2")
    
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" in page:
            for obj in page["Contents"]:
                key = obj["Key"]
                # Apply additional filters if needed
                parts = key.replace(f"{s3_prefix}/", "").split("/")
                if model and len(parts) > 3 and parts[2] != model:
                    continue
                results.append(key)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload outputs directory to S3")
    parser.add_argument(
        "action",
        choices=["upload", "list"],
        help="Action to perform: upload files or list existing files",
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default="outputs",
        help="Local path to outputs directory",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        required=True,
        help="S3 bucket name",
    )
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default="outputs",
        help="Prefix for S3 keys (default: outputs)",
    )
    parser.add_argument(
        "--competition",
        type=str,
        help="Optional competition name to filter (e.g., aime)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        help="Optional provider name to filter (e.g., bedrock, openrouter)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Optional model name to filter",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="AWS region (default: us-east-1)",
    )
    parser.add_argument(
        "--endpoint-url",
        type=str,
        help="Optional S3 endpoint URL (for S3-compatible services)",
    )

    args = parser.parse_args()

    if args.action == "upload":
        upload_outputs_to_s3(
            outputs_dir=args.outputs_dir,
            bucket=args.bucket,
            s3_prefix=args.s3_prefix,
            competition=args.competition,
            provider=args.provider,
            model=args.model,
            region=args.region,
            endpoint_url=args.endpoint_url,
        )
    elif args.action == "list":
        files = list_outputs_on_s3(
            bucket=args.bucket,
            s3_prefix=args.s3_prefix,
            competition=args.competition,
            provider=args.provider,
            model=args.model,
            region=args.region,
            endpoint_url=args.endpoint_url,
        )
        logger.info(f"Found {len(files)} files in S3")
        for file in files[:20]:  # Show first 20
            logger.info(f"  {file}")
        if len(files) > 20:
            logger.info(f"  ... and {len(files) - 20} more")

