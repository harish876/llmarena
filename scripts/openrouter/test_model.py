import argparse
import os
import sys

import yaml

# Ensure `src/` is on sys.path like other scripts
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from matharena.api_client import APIClient


def load_model_config(config_path: str) -> dict:
    """Loads a model configuration YAML file.
    
    Args:
        config_path: Path to the config file, can be relative to configs/models/ or absolute.
    
    Returns:
        dict: The configuration dictionary.
    """
    # If path doesn't exist, try relative to configs/models/
    if not os.path.isabs(config_path) and not os.path.exists(config_path):
        # Try openrouter first
        test_path = os.path.join(ROOT_DIR, "configs", "models", "openrouter", config_path)
        if os.path.exists(test_path):
            config_path = test_path
        else:
            # Try general models directory
            config_path = os.path.join(ROOT_DIR, "configs", "models", config_path)
    
    # Try adding .yaml extension if needed
    if not os.path.exists(config_path):
        if os.path.exists(config_path + ".yaml"):
            config_path = config_path + ".yaml"
        elif os.path.exists(config_path + ".yml"):
            config_path = config_path + ".yml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    if not config:
        raise ValueError(f"Model config file is empty: {config_path}")
    
    if "model" not in config:
        raise ValueError(f"Model config file must specify 'model': {config_path}")
    
    return config


def load_prompt(prompt_path: str) -> str:
    """Loads a prompt from a file (YAML with 'prompt' key or plain text).
    
    Args:
        prompt_path: Path to the prompt file.
    
    Returns:
        str: The prompt content.
    """
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    with open(prompt_path, "r") as f:
        content = f.read().strip()
    
    if not content:
        raise ValueError(f"Prompt file is empty: {prompt_path}")
    
    # Try parsing as YAML first (in case it has a 'prompt' key)
    try:
        yaml_content = yaml.safe_load(content)
        if isinstance(yaml_content, dict) and "prompt" in yaml_content:
            return str(yaml_content["prompt"])
    except:
        pass
    
    # Otherwise treat as plain text
    return content


def create_api_client_from_config(model_config: dict) -> APIClient:
    """Creates an APIClient instance from a model configuration.
    
    Args:
        model_config: Model configuration dictionary.
    
    Returns:
        APIClient: Configured API client instance.
    """
    model = model_config.get("model")
    api = model_config.get("api", "openrouter")
    
    # Extract kwargs for APIClient
    api_client_kwargs = {
        "model": model,
        "api": api,
    }
    
    # Add common parameters
    for key in ["max_tokens", "timeout", "temperature", "top_p", "top_k", 
                "batch_processing", "use_openai_responses_api", 
                "read_cost", "write_cost", "concurrent_requests",
                "max_retries", "reasoning_effort"]:
        if key in model_config and model_config[key] is not None:
            api_client_kwargs[key] = model_config[key]
    
    # Add any other kwargs from config (excluding metadata fields)
    metadata_keys = ["human_readable_id", "date", "model", "api"]
    for key, value in model_config.items():
        if key not in metadata_keys and key not in api_client_kwargs and value is not None:
            api_client_kwargs[key] = value
    
    return APIClient(**api_client_kwargs)


def execute_query(model_config_path: str, prompt_path: str | None, prompt_text: str | None) -> None:
    """Loads a model config and executes a single query.
    
    Args:
        model_config_path: Path to the model config file.
        prompt_path: Optional path to a prompt config file.
        prompt_text: Optional prompt text (overrides prompt_path if provided).
    """
    # Load model config
    model_config = load_model_config(model_config_path)
    api_name = model_config.get("api", "unknown")
    
    # Check API key
    if api_name == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    elif api_name == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable not set")
    elif api_name == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")
    
    # Load prompt
    if prompt_text:
        prompt = prompt_text
    elif prompt_path:
        prompt = load_prompt(prompt_path)
    else:
        prompt = "Reply with exactly the single word: pong"
    
    print(f"=== Executing query ===")
    print(f"Model config: {model_config_path}")
    print(f"Model: {model_config.get('model')}")
    print(f"API: {api_name}")
    if prompt_path:
        print(f"Prompt file: {prompt_path}")
    print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    print()
    
    # Create client and execute
    client = create_api_client_from_config(model_config)
    queries = [[{"role": "user", "content": prompt}]]
    
    try:
        idx, conversation, detailed_cost = next(iter(client.run_queries(queries, no_tqdm=True)))
        
        assistant_msgs = [
            m for m in conversation if m.get("role") == "assistant" and isinstance(m.get("content"), str)
        ]
                
        response = assistant_msgs[-1]["content"]
        print(f"Response:\n{response}")
        print()
        
        # Print token usage if available
        if detailed_cost:
            print(f"Token usage:")
            print(f"  Input tokens: {detailed_cost.get('input_tokens', 'N/A')}")
            print(f"  Output tokens: {detailed_cost.get('output_tokens', 'N/A')}")
            if 'cost' in detailed_cost:
                print(f"  Cost: ${detailed_cost['cost']:.6f}")
        
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load a model config and execute a single query using APIClient"
    )
    parser.add_argument(
        "model_config",
        type=str,
        help="Path to model config file (from configs/models/ or absolute path)"
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Path to prompt config file (YAML with 'prompt' key or plain text)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt text (overrides --prompt-file if provided)"
    )
    
    args = parser.parse_args()
    
    execute_query(args.model_config, args.prompt_file, args.prompt)

