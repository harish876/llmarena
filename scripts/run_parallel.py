import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger

from matharena.runner import Runner

# Main args: which competition to run with which models; how many runs per problem
parser = argparse.ArgumentParser()
parser.add_argument("--comp", type=str, required=True, help="Competition config to run")
parser.add_argument(
    "--models",
    type=str,
    nargs="+",
    required=True,
    help="List of model configs to run, might have scaffolding, example: xai/grok-4",
)
parser.add_argument("--n", type=int, default=4, help="Number of runs per problem")
parser.add_argument("--max-problems", type=int, default=None, help="Maximum number of problems to run (useful for trial runs)")

# skip-existing is default
parser.add_argument(
    "--redo-all", action="store_true", help="Redo all (model, problem) pairs regardless of existing runs"
)

# Generally ok to keep defaults here
parser.add_argument("--comp-configs-dir", type=str, default="configs/competitions")
parser.add_argument("--model-configs-dir", type=str, default="configs/models")
parser.add_argument("--output-dir", type=str, default="outputs")
parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of parallel model runs")
args = parser.parse_args()

def run_model(runner: Runner, model):
    """Run a single model with error handling."""
    try:
        logger.info(f"Starting model: {model}")
        runner.run(model)
        logger.info(f"Completed model: {model}")
        return model, True, None
    except Exception as e:
        logger.error(f"Error running model {model}: {e}")
        return model, False, str(e)

logger.info(f"Initializing runner for competition {args.comp}")
runner = Runner(args.comp, args.n, args.comp_configs_dir, args.model_configs_dir, args.output_dir, args.redo_all, max_problems=args.max_problems)

logger.info(f"Running {len(args.models)} models with max {args.max_workers} workers")
with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
    future_to_model = {
        executor.submit(run_model, runner, model): model 
        for model in args.models
    }
    completed_models = []
    failed_models = []
    
    for future in as_completed(future_to_model):
        model = future_to_model[future]
        try:
            model_name, success, error = future.result()
            if success:
                completed_models.append(model_name)
            else:
                failed_models.append((model_name, error))
        except Exception as e:
            logger.error(f"Unexpected error for model {model}: {e}")
            failed_models.append((model, str(e)))

logger.info(f"Completed models: {completed_models}")
if failed_models:
    logger.error(f"Failed models: {[f'{model}: {error}' for model, error in failed_models]}")
