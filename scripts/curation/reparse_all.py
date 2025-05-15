import argparse
import os
from matharena.runner import run
import yaml
from loguru import logger

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=4)
parser.add_argument("--comp", type=str, nargs="+", required=True)
parser.add_argument("--output-folder", type=str, default="outputs")
parser.add_argument("--configs-folder", type=str, default="configs/models")
parser.add_argument("--competition-config-folder", type=str, default="configs/competitions")
args = parser.parse_args()

for comp in args.comp:
    comp_path = comp.replace('.yaml', '')
    comp_output_path = os.path.join(args.output_folder, comp_path)
    comp_path = comp_path + '.yaml'
    for api in os.listdir(comp_output_path):
        for model in os.listdir(os.path.join(comp_output_path, api)):
            config_path = f'{api}/{model}' 
            config_path = config_path + ".yaml" if not config_path.endswith(".yaml") else config_path
            with open(f"{args.configs_folder}/{config_path}", 'r') as f:
                model_config = yaml.safe_load(f)
            
            model_config["n"] = model_config.get("n", args.n)
            logger.info(f"Reparsing model {config_path} for {comp}")

            run(model_config, config_path, comp, skip_existing=True, skip_all=True,
                output_folder=args.output_folder, competition_config_folder=args.competition_config_folder)