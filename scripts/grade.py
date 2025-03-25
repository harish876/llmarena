import argparse
import os
from matharena.runner import run
from matharena.grader import run_grader
import yaml
from loguru import logger

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=4)
parser.add_argument("--grader_config", type=str, required=True)
parser.add_argument("--solver_config", type=str, nargs="+", required=True)
parser.add_argument("--skip_existing", action="store_true")
parser.add_argument("--comp", type=str, required=True)
parser.add_argument("--output-folder", type=str, default="outputs")
parser.add_argument("--grading-folder", type=str, default="autogradings")
parser.add_argument("--configs-folder", type=str, default="configs")
args = parser.parse_args()

with open(f"{args.configs_folder}/{args.grader_config}", 'r') as f:
    grader_config = yaml.safe_load(f)


for solver_config_path in args.solver_config:
    with open(f"{args.configs_folder}/{solver_config_path}", 'r') as f:
        solver_config = yaml.safe_load(f)
    logger.info(f"Running grader {args.grader_config} on config: {solver_config}")
    grader_config["n"] = solver_config.get("n", args.n)

    run_grader(grader_config, solver_config_path, args.comp, skip_existing=args.skip_existing,
                output_folder=args.output_folder, grading_folder=args.grading_folder)
