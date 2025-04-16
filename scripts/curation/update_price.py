import json
import argparse
import os
import yaml


def update_price(config_path, comp):
    path_output_folder = os.path.join(args.output_folder, comp, config_path.replace(".yaml", ""))

    all_files = os.listdir(path_output_folder)

    model_config = os.path.join(args.config_folder, config_path)
    model_config = yaml.safe_load(open(model_config, "r"))
    input_price = model_config["read_cost"]
    output_price = model_config["write_cost"]

    for file in all_files:
        if file.endswith(".json"):
            file_path = os.path.join(path_output_folder, file)
            with open(file_path, "r") as f:
                data = json.load(f)

            # Update the price in the JSON data
            data["cost"]["cost"] = input_price * data["cost"]["input_tokens"] + output_price * data["cost"]["output_tokens"]
            data["cost"]["cost"] /= 10 ** 6

            for i in range(len(data["detailed_costs"])):
                data["detailed_costs"][i]["cost"] = input_price * data["detailed_costs"][i]["input_tokens"] + output_price * data["detailed_costs"][i]["output_tokens"]
                data["detailed_costs"][i]["cost"] /= 10 ** 6
            # Write the updated data back to the file
            with open(file_path, "w") as f:
                json.dump(data, f)
            

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to the config file")
parser.add_argument("--comp", type=str, required=True, help="Path to the comparison file")
parser.add_argument("--output-folder", type=str, default="outputs", help="Path to the output folder")
parser.add_argument("--config-folder", type=str, default="configs/models", help="Path to the config folder")
args = parser.parse_args()

args.config = args.config + ".yaml" if not args.config.endswith(".yaml") else args.config
update_price(args.config, args.comp)
