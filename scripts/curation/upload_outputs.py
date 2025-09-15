from datasets import Dataset
import pandas as pd
import os
import json
from matharena.configs import load_configs
from loguru import logger
import yaml



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Upload dataset to Hugging Face Hub")
    parser.add_argument("--org", type=str, default="MathArena", help="Hugging Face organization name")
    parser.add_argument("--repo-name", type=str, help="Hugging Face repo name", required=True)
    parser.add_argument("--comp", type=str, help="Competition name", required=True)
    parser.add_argument("--output-folder", type=str, default="outputs", help="Output directory for the dataset")
    parser.add_argument("--configs-folder", type=str, default="configs/models", help="Directory containing the configs of the models")
    parser.add_argument("--competition-configs-folder", type=str, default="configs/competitions", help="Directory containing the configs")
    parser.add_argument("--public", action="store_true", help="Make the dataset public (not advised, best to keep it private and manually share)")

    args = parser.parse_args()

    folder = os.path.join(args.output_folder, args.comp)

    configs = load_configs(args.configs_folder)

    competition_config = yaml.safe_load(open(os.path.join(args.competition_configs_folder, args.comp + ".yaml"), "r"))

    all_data = []

    for config_path in configs:
        folder_model = os.path.join(folder, config_path)
        if not os.path.exists(folder_model):
            logger.info(f"Folder {folder_model} does not exist, skipping...")
            continue

        if "model_config" in configs[config_path]:
            config_model = configs[configs[config_path]["model_config"].replace("models/", "")]
            configs[config_path]["read_cost"] = config_model["read_cost"]
            configs[config_path]["write_cost"] = config_model["write_cost"]
            
        # list all files in the folder
        problem_files = os.listdir(folder_model)
        for file in problem_files:
            file_name = os.path.join(folder_model, file)
            problem_name = file.split(".")[0]
            data = json.load(open(file_name, "r"))
            gold_answer = data["gold_answer"]
            problem = data["problem"]
            for i in range(len(data["messages"])):
                default_dict = {
                    "problem_idx": problem_name,
                    "problem": problem,
                    "model_name": configs[config_path]["human_readable_id"],
                    "model_config": config_path,
                    "idx_answer": i,
                    "user_message": data["messages"][i][0]["content"],
                    "answer": data["messages"][i][-1]["content"],
                    "input_tokens": data["detailed_costs"][i]["input_tokens"],
                    "output_tokens": data["detailed_costs"][i]["output_tokens"],
                    "cost": data["detailed_costs"][i]["cost"],
                    "input_cost_per_tokens": configs[config_path]["read_cost"],
                    "output_cost_per_tokens": configs[config_path]["write_cost"],
                }

                if "source" in data:
                    default_dict["source"] = data["source"]

                if competition_config.get("final_answer", True):
                    extra_dict = {
                        "gold_answer": gold_answer,
                        "parsed_answer": data["answers"][i],
                        "correct": data["correct"][i],
                    }
                else:
                    extra_dict = dict()
                    for j in range(len(data["judgment"])):
                        extra_dict_judge = {
                            f"points_judge_{j+1}": data["judgment"][j][i]["points"],
                            f"grading_details_judge_{j+1}": data["judgment"][j][i]["details"],
                            f"error_judge_{j+1}": data["judgment"][j][i]["error"] if "error" in data["judgment"][j][i] else None,
                            f"max_points_judge_{j+1}": data["judgment"][j][i]["max_points"],
                        }
                        extra_dict.update(extra_dict_judge)


                all_data.append({**default_dict, **extra_dict})
    df = pd.DataFrame(all_data)
    if "parsed_answer" in df.columns:
        df["parsed_answer"] = df["parsed_answer"].astype(str)
    if "source" in df.columns:
        df = df[df["source"].apply(lambda x: "smt" not in x.lower())]
    
    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub(
        os.path.join(args.org, args.repo_name),
        private=not args.public,
    )




