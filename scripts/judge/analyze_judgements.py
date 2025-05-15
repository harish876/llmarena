import argparse
import os
import json
import numpy as np
import pandas as pd
import glob
import re


parser = argparse.ArgumentParser()
parser.add_argument("--comp", default="usamo/usamo_2024")
args = parser.parse_args()

# go over all folders in judgments/{args.comp} recursively. Find all files ending on .json

def get_files(path):
    glob_path = os.path.join(path, "**", "*.json")
    return glob.glob(glob_path, recursive=True)

judgments = {}

for base_folder in ['grading', 'autogradings']:
    judgments_files = get_files(f"{base_folder}/{args.comp}")
    outputs = get_files(f"outputs/{args.comp}")

    grading_scheme = json.load(open(f"data/{args.comp}/grading_scheme.json"))
    concat_data = []
    for output in outputs:
        file_content = json.load(open(output))

        anon_id = file_content["anonymous_id"]
        grading_scheme_question = grading_scheme[file_content["idx"]]

        file_content["judgment"] = []
        file_content["correct"] = []
        for judgment_file in judgments_files:
            anon_id_judgment = judgment_file.replace('\\','/').split("/")[-1].replace(".json", "")
            if "_" in anon_id_judgment:
                anon_id_judgment = anon_id_judgment.split("_")[0]
            # this is the correct judgment file associated with this answer
            if anon_id == anon_id_judgment:
                judgment_all = json.load(open(judgment_file))

                assert len(judgment_all) == len(file_content["messages"]), f"Judgment file {judgment_file} does not have enough judgments."
                
                # for each of the 4 iterations of a model
                for i, judgment_content in enumerate(judgment_all):
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
                    
                    grader = judgment_file.replace('\\','/').split("/")[-1].replace(".json", "")
                    try:
                        grader = int(grader.split('_')[-1])
                    except:
                        grader = '-'.join(grader.split('_')[-1].split('-')[:-1])
                    concat_data.append({
                        'id': anon_id,
                        'problem': file_content["idx"],
                        'model_name': output.replace('\\', '/').split('/')[-2],
                        'points': judgment_content["points"],
                        'correct': judgment_content["points"] >=6, 
                        'error': judgment_content["error"] if "error" in judgment_content else None,
                        'cost': file_content['cost']['cost'],
                        'gen': i,
                        'grader': grader
                    })
                file_content["judgment"].append(judgment_all)
                points = [judgment_content["points"] / judgment_content["max_points"] for judgment_content in judgment_all]
                file_content["correct"].append(points)
    judgments[base_folder] = concat_data
            
grading_scheme = json.load(open(f"data/{args.comp}/grading_scheme.json"))

auto_data = pd.DataFrame(judgments['autogradings'])
data = pd.DataFrame(judgments['grading'])

auto_data['points'] = auto_data.groupby(['model_name', 'problem', 'gen']).points.transform(lambda x: x.fillna(x[~x.isna()].mean()))
print(auto_data.groupby(['model_name', 'problem', 'grader']).points.mean().reset_index().groupby(['model_name', 'grader']).points.sum())

# res_per_problem_auto = auto_data.groupby(['model_name', 'problem']).points.mean()
# res_per_problem = data.groupby(['model_name', 'problem']).points.mean()
breakpoint()



model_map_res ={
    'claude-37': "\\claude",
    'qwq': '\\qwq',
    'deepseek_r1': '\\rone',
    'gemini-flash-thinking-2.0': '\\flthink',
    'o3-mini': '\\othree',
    'o1-pro': '\\oone',
}

model_map_fig ={
    'claude-37': "\\textsc{Claude 3.7}",
    'qwq': '\\textsc{QwQ}',
    'deepseek_r1': '\\textsc{R1}',
    'gemini-flash-thinking-2.0': '\\textsc{Flash-Thinking}',
    'o3-mini': '\\textsc{o3-mini}',
    'o1-pro': '\\textsc{o1 Pro}',
}

problems = sorted(data["problem"].unique())
models = sorted(data["model_name"].unique())

cost = data.fillna(0).groupby(["model_name", "problem"])["cost"].agg(["mean"]).reset_index()
cost = cost.pivot(index="model_name", columns="problem", values="mean").fillna(0)

summary = data.groupby(["model_name", "problem"])["points"].agg(["mean", "count"]).reset_index()
summary = summary.pivot(index="model_name", columns="problem", values="mean").fillna(0)

summary_correct = data.groupby(["model_name", "problem"])["correct"].agg(["sum"]).reset_index()
summary_correct = summary_correct.pivot(index="model_name", columns="problem", values="sum").fillna(0)

summary["Total"] = summary.sum(axis=1)
summary = summary.sort_values(by="Total", ascending=False)

summary_correct["Total"] = (summary_correct > 0).sum(axis=1)
cost["Total"] = cost.sum(axis=1)

latex_str = r"""\begin{table}[!hbt]
    \centering
    \caption{Main results of our evaluation. We measure cost in USD, and report the average score across all generations and graders for each problem. We also report in brackets the number of fully solved problems.}
    \vspace{-1mm}
    \begin{tabular}{l""" + "l" * len(problems) + "ll}\n"
latex_str += r"        \toprule" + "\n"
latex_str += r"        \textbf{Model} & " + " & ".join([f"\\textbf{{{p}}}" for p in problems]) + " & \\textbf{Total} & \\textbf{Cost} \\\\\n"
latex_str += r"        \midrule" + "\n"

for model in summary.index:
    row = f"        {model_map_res[model]} & "
    row += " & ".join(f"{summary.loc[model, p]:.1f}/7 ({summary_correct.loc[model, p]//2})" for p in problems)
    row += f" & {summary.loc[model, 'Total']:.1f}/42 ({summary_correct.loc[model, 'Total']}) & {cost.loc[model, 'Total']:.2f} \\\\\n"
    latex_str += row

latex_str += r"        \bottomrule" + "\n"
latex_str += r"    \end{tabular}" + "\n"
latex_str += r"    \label{tab:main_results}" + "\n"
latex_str += r"\end{table}"

print(latex_str)


import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Helvetica"
# })
rcParams['text.usetex'] = True
plt.rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'

error_counts = data.groupby(["error", "model_name"]).size().unstack(fill_value=0)

# Sort error types by total frequency
error_counts = error_counts.loc[error_counts.sum(axis=1).sort_values(ascending=False).index]

print(models)
# Define a color palette (you can modify this)
custom_colors = {
    'claude-37': '#f3e9d7',  
    'deepseek_r1': '#4f6bfe',
    'gemini-flash-thinking-2.0': '#21437a',
    'o1-pro': '#70a597',
    'o3-mini': '#a0d5c7',
    'qwq': '#ff8620'
}

label_map = {
    "idea": "Creativity",
    "logic": "Logic",
    "algebra": "Algebra/Arithmetics",
    "assumption": "Assumption"
}

# Plot stacked bar chart
fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

bottom = None
for model in sorted(error_counts.columns):  # Ensure models are sorted
    ax.bar([label_map[x] for x in error_counts.index], error_counts[model], label=model_map_fig[model], bottom=bottom, color=custom_colors.get(model, "gray"))
    bottom = error_counts[model] if bottom is None else bottom + error_counts[model]

# Labels and legend
ax.set_xlabel("Error Type")
ax.set_ylabel("Frequency")
ax.set_title("Stacked Error Distribution by Model")
ax.legend(title="Model")

sns.despine(left=True, bottom=True)
ax.set_facecolor((0.97,0.97,0.97))

plt.tight_layout()
plt.savefig('test.png')

    

