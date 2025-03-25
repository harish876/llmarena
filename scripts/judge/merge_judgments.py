import argparse
import os
import json
import numpy as np
import glob
from loguru import logger

parser = argparse.ArgumentParser()
parser.add_argument("--comp", default="usamo/usamo_2024")
parser.add_argument("--grading-folder", default="grading")
args = parser.parse_args()

# go over all folders in judgments/{args.comp} recursively. Find all files ending on .json

def get_files(path):
    glob_path = os.path.join(path, "**", "*.json")
    return glob.glob(glob_path, recursive=True)

judgments_files = get_files(f"{args.grading_folder}/{args.comp}")
outputs = get_files(f"outputs/{args.comp}")

grading_scheme = json.load(open(f"data/{args.comp}/grading_scheme.json"))

for output in outputs:
    file_content = json.load(open(output))
    anon_id = file_content["anonymous_id"]
    grading_scheme_question = grading_scheme[file_content["idx"]]

    file_content["judgment"] = []
    file_content["correct"] = []

    for judgment_file in judgments_files:
        anon_id_judgment = judgment_file.replace("\\", "/").split("/")[-1].replace(".json", "")
        if "_" in anon_id_judgment:
            anon_id_judgment = anon_id_judgment.split("_")[0]
        # this is the correct judgment file associated with this answer
        if anon_id == anon_id_judgment:
            judgment_all = json.load(open(judgment_file))

            assert len(judgment_all) == len(file_content["messages"]), f"Judgment file {judgment_file} does not have enough judgments."
            
             # for each of the 4 iterations of a model
            for judgment_content in judgment_all:
                judgment_content["max_points"] = grading_scheme_question["points"]
                assert "details" in judgment_content, f"Judgment file {judgment_file} does not have the correct format."
                assert sum([judgment_part["points"] for judgment_part in judgment_content["details"]]) == judgment_content["points"], f"Judgment file {judgment_file} has incorrect points."
                assert len(judgment_content["details"]) == len(grading_scheme_question["scheme"]), f"Judgment file {judgment_file} does not have the correct format."
                # Find the judgment parts in the grading scheme and update the points
                for index, grading_scheme_part in enumerate(grading_scheme_question["scheme"]):
                    is_found = False
                    index_judgment = index
                    if all("title" in judgment_part for judgment_part in judgment_content["details"]):
                        for index_j, judgment_part in enumerate(judgment_content["details"]):
                            assert "points" in judgment_part and "desc" in judgment_part, f"Judgment file {judgment_file} does not have the correct format."
                            if judgment_part["title"] == grading_scheme_part["title"]:
                                index_judgment = index_j
                                break
                    judgment_part = judgment_content["details"][index_judgment]
                    is_found = True
                    judgment_part["max_points"] = grading_scheme_part["points"]
                    assert judgment_part["points"] <= grading_scheme_part["points"], f"Judgment part with title {judgment_part['title']} has more points than allowed in grading scheme for file {judgment_file}"
                    judgment_part["grading_scheme_desc"] = grading_scheme_part["desc"]
                    if not is_found:
                        raise ValueError(f"Judgment part with title {grading_scheme_part['title']} not found in grading scheme for file {judgment_file}. You should include all points, even those that are 0.")
            
            file_content["judgment"].append(judgment_all)
            points = [judgment_content["points"] / judgment_content["max_points"] for judgment_content in judgment_all]
            file_content["correct"].append(points)
    
    if len(file_content["judgment"]) == 0:
        logger.info(f"No judgment file found for {output}")
        continue

    file_content["correct"] = list(np.mean(file_content["correct"], axis=0))
    file_content["pass_at_1"] = np.mean(file_content["correct"])

    with open(output, "w") as f:
        json.dump(file_content, f)
