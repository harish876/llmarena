import argparse
import os
import sys
from typing import List

# Ensure `src/` is on sys.path like other scripts
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    import boto3 
except Exception:
    boto3 = None

from matharena.api_client import APIClient


def have_aws_credentials(profile: str | None) -> bool:
    if boto3 is None:
        return False
    try:
        session = boto3.Session(profile_name=profile) if profile else boto3.Session()
        return session.get_credentials() is not None
    except Exception:
        return False


def list_foundation_models(region: str, profile: str | None) -> None:
    if boto3 is None:
        raise RuntimeError("boto3 not installed; cannot list Bedrock foundation models")
    if not have_aws_credentials(profile):
        raise RuntimeError("AWS credentials not available; set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY and region")

    session = (
        boto3.Session(profile_name=profile, region_name=region)
        if profile
        else boto3.Session(region_name=region)
    )
    client = session.client("bedrock")
    resp = client.list_foundation_models()
    summaries = resp.get("modelSummaries", [])
    if not summaries:
        raise RuntimeError("No foundation models returned by Bedrock")

    for s in summaries:
        modalities = ",".join((s.get("outputModalities") or []))
        print(f"{s.get('modelId')} | provider={s.get('providerName')} | modality={modalities}")


def smoke_test_models(models: List[str], timeout: int, max_tokens: int, region: str, profile: str | None) -> None:
    if boto3 is None:
        raise RuntimeError("boto3 not installed; cannot run Bedrock smoke test")
    if not have_aws_credentials(profile):
        raise RuntimeError("AWS credentials not available; set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY and region")
    if not models:
        raise RuntimeError("No models provided. Use --models model1,model2 or --models-file path.txt")

    prompt = "Reply with exactly the single word: pong"

    for model in models:
        print(f"\n=== Smoke testing model: {model} ===")
        client = APIClient(
            model=model,
            api="bedrock",
            max_tokens=max_tokens,
            timeout=timeout,
            temperature=0,
        )

        queries = [[{"role": "user", "content": prompt}]]
        idx, conversation, detailed_cost = next(iter(client.run_queries(queries, no_tqdm=True)))

        assistant_msgs = [
            m for m in conversation if m.get("role") == "assistant" and isinstance(m.get("content"), str)
        ]
        if not assistant_msgs:
            raise RuntimeError(f"No assistant message returned for model {model}")
        last = assistant_msgs[-1]["content"].strip().lower()
        if "pong" not in last:
            raise RuntimeError(f"Unexpected response for {model}: {last}")
        print("OK: response contains 'pong'")


def load_models_arg(models_arg: str | None, models_file: str | None) -> List[str]:
    models: List[str] = []
    if models_arg:
        models.extend([m.strip() for m in models_arg.split(",") if m.strip()])
    if models_file:
        with open(models_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    models.append(line)
    # de-dup
    return list(dict.fromkeys(models))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bedrock utilities: list foundation models and smoke test models")
    parser.add_argument("--list", action="store_true", help="List foundation models")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test against provided models")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model IDs for smoke test")
    parser.add_argument("--models-file", type=str, default=None, help="File with one model ID per line")
    parser.add_argument("--region", type=str, default=os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1")))
    parser.add_argument("--profile", type=str, default=os.getenv("AWS_PROFILE"))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=64)

    args = parser.parse_args()

    if not args.list and not args.smoke:
        parser.error("Specify at least one of --list or --smoke")

    if args.list:
        list_foundation_models(args.region, args.profile)

    if args.smoke:
        models = load_models_arg(args.models, args.models_file)
        smoke_test_models(models, args.timeout, args.max_tokens, args.region, args.profile)


