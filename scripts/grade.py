import argparse
import os
from matharena.grader import run_grader
import yaml
from loguru import logger

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=4)
parser.add_argument("--grader-config", type=str, required=True)
parser.add_argument("--solver-config", type=str, nargs="+", required=True)
parser.add_argument("--skip-existing", action="store_true")
parser.add_argument("--comp", type=str, required=True)
parser.add_argument("--output-folder", type=str, default="outputs")
parser.add_argument("--grading-folder", type=str, default="autogradings")
parser.add_argument("--configs-folder", type=str, default="configs/models")
parser.add_argument("--competition-config-folder", type=str, default="configs/competitions")
parser.add_argument("--grading-config", type=str, default="configs/autograding/config.yaml")
args = parser.parse_args()


args.grader_config = args.grader_config + ".yaml" if not args.grader_config.endswith(".yaml") else args.grader_config
args.solver_config = [f"{config}.yaml" if not config.endswith('.yaml') else config for config in args.solver_config]
with open(f"{args.configs_folder}/{args.grader_config}", 'r') as f:
    grader_config = yaml.safe_load(f)


for solver_config_path in args.solver_config:
    with open(f"{args.configs_folder}/{solver_config_path}", 'r') as f:
        solver_config = yaml.safe_load(f)
    logger.info(f"Running grader {args.grader_config} on config: {solver_config}")
    grader_config["n"] = solver_config.get("n", args.n)

    run_grader(grader_config, solver_config_path, args.comp, skip_existing=args.skip_existing,
                output_folder=args.output_folder, grading_folder=args.grading_folder, 
                competition_config_folder=args.competition_config_folder, 
                autograding_config_path=args.grading_config)
