# Usage: uv run scripts/paper/timeline.py --comp hmmt

import json
import os
import yaml
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import argparse
import re
from matplotlib import rcParams

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Helvetica"
# })
# rcParams['text.usetex'] = True
# plt.rcParams['text.usetex'] = True
# rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'

parser = argparse.ArgumentParser()
parser.add_argument("--comps", type=str, nargs="+", default="aime")
parser.add_argument("--cost", action="store_true", help="Use cost instead of score")
parser.add_argument("--save-file", type=str, default="paper_data/timeline.pdf")
args = parser.parse_args()

models_dir = "configs/models/"
model_dates = {}
for org in os.listdir(models_dir):
    org_dir = os.path.join(models_dir, org)
    if not os.path.isdir(org_dir):
        continue
        
    for model_file in os.listdir(org_dir):
        if not model_file.endswith('.yaml'):
            continue
        model_path = os.path.join(org_dir, model_file)
        with open(model_path, 'r') as f:
            model_config = yaml.safe_load(f)
            date = model_config.get("date", None) # 2025-04-09
            if date is None:
                continue
            model_dates[model_config["human_readable_id"]] = date
            # date = datetime.strptime(date, "%Y-%m-%d") # Keep as string for now
            # print(date)

comp_locations = []

for comp in args.comps:
    if comp == "aime":
        results_path = "website/flaskr/static/data/aime/aime_2025/results.json"
    elif comp == "hmmt":
        results_path = "website/flaskr/static/data/hmmt/hmmt_feb_2025/results.json"
    elif comp == "smt":
        results_path = "website/flaskr/static/data/smt/smt_2025/results.json"
    elif comp == "brumo":
        results_path = "website/flaskr/static/data/brumo/brumo_2025/results.json"
    else:
        raise ValueError(f"Invalid competition: {comp}")
    comp_locations.append(results_path)

model_res, model_avg, model_cost = {}, {}, {}

for results_path in comp_locations:
    with open(results_path, "r") as f:
        results = json.load(f)
        for result in results:
            for model in result:
                if model == "question":
                    continue
                if result["question"] == "Avg":
                    model_avg[model] = model_avg.get(model, []) + [result[model]]
                elif result["question"] == "Cost" and result[model] != "N/A":
                    model_cost[model] = model_cost.get(model, []) + [result[model] / 4]
                else:
                    if model not in model_res:
                        model_res[model] = []
                    model_res[model].append(result[model])

models_to_remove = []
for model in model_avg:
    if model not in model_cost:
        models_to_remove.append(model)
        continue
    if len(model_avg[model]) != len(args.comps) or len(model_cost[model]) != len(args.comps):
        models_to_remove.append(model)
    model_avg[model] = sum(model_avg[model]) / len(model_avg[model])
    model_cost[model] = sum(model_cost[model]) / len(comp_locations)

for model in models_to_remove:
    del model_avg[model]
    if model in model_cost:
        del model_cost[model]

# print(model_avg)
# for model in model_res:
    # print(model, model_dates[model], model_avg[model])

# Prepare data for plotting
plot_dates = []
plot_scores = []
plot_labels = []

model_cost = {k: v for k, v in model_cost.items() if v != "N/A"}

instersection_keys = set(model_dates.keys()).intersection(set(model_avg.keys())).intersection(set(model_cost.keys()))

if args.cost:
    model_dates = model_cost

# remove models that are not in both
model_dates = {k: model_dates[k] for k in instersection_keys}
model_avg = {k: model_avg[k] for k in instersection_keys}

for model, avg_score in model_avg.items():
    if model in model_dates:
        date_str = model_dates[model]
        try:
            if not args.cost:
                plot_dates.append(datetime.strptime(date_str, "%Y-%m-%d"))
            else:
                plot_dates.append(date_str)
            plot_scores.append(avg_score)
            plot_labels.append(model)
        except ValueError:
            print(f"Warning: Could not parse date '{date_str}' for model '{model}'. Skipping.")
    else:
        print(f"Warning: Date not found for model '{model}'. Skipping.")

pareto, pareto_names = [], set()
for model, avg_score in model_avg.items():
    is_pareto = True
    for other_model, other_score in model_avg.items():
        if other_score > avg_score and model_dates[other_model] <= model_dates[model]:
            is_pareto = False
            break
    if is_pareto:
        pareto.append((model_dates[model], avg_score))
        pareto_names.add(model)
pareto = sorted(pareto, key=lambda x: x[0])
if not args.cost:
    pareto_dates = [datetime.strptime(x[0], "%Y-%m-%d") for x in pareto]
else:
    pareto_dates = [x[0] for x in pareto]
pareto_scores = [x[1] for x in pareto]


# Set the seaborn style and context for better aesthetics
plt.rcParams.update({'font.size': 15})


# Create a DataFrame for easier plotting with seaborn
import pandas as pd
df = pd.DataFrame({
    'date': plot_dates,
    'score': plot_scores,
    'model': plot_labels
})

# Sort by date for better visualization
df = df.sort_values('date')

# Create the plot
fig, ax = plt.subplots(figsize=(6, 4))

ax.scatter(plot_dates, plot_scores, s=60)

# Plot Pareto Frontier
ax.plot(pareto_dates, pareto_scores, marker='o', linestyle='-', color='red', label='Pareto Frontier', linewidth=1, 
         markersize=8)

# Add labels to points
for i, label in enumerate(plot_labels):
    if label in pareto_names:
        offset_x = 0
        if i % 2 == 1:
            offset_y = 10
        else:
            offset_y = -16
        if label == "gpt-4o":
            offset_x = 24
            offset_y = -3
        if args.comps[0] == "hmmt" and not args.cost:
            if "gemini-2.5-pro" in label:
                offset_x = -45
                offset_y = 0
            elif "o4-mini" in label or "o3" in label:
                offset_y = -15
                offset_x = 0
        if args.cost and len(args.comps) == 4:
            if "14B" in label:
                offset_x = 68
                offset_y = -2
            elif "Grok 3 Mini (low)" in label:
                offset_y = -15
                offset_x = 2
            elif "Grok 3 Mini (high)" in label:
                offset_y = 10
                offset_x = -6
            elif "o4-mini" in label:
                offset_y = -15
                offset_x = 0
            elif "A22B" in label:
                offset_x = 36
                offset_y = -10
            elif "A3B" in label:
                offset_x = 35
                offset_y = -2
            elif "o3" in label:
                offset_y = -15
                offset_x = 0
        # remove everything within brackets
        if not (args.cost and len(args.comps) == 4 and "Grok" in label):
            label = re.sub(r'\s*\(.*?\)', '', label)
        label = label.replace("-A22B", "")
        label = label.replace("-A3B", "")
        ax.annotate(label, (plot_dates[i], plot_scores[i]), textcoords="offset points", 
                    xytext=(offset_x,offset_y), ha='center', fontsize=10)

comp_dates = {
    "aime": "2025-02-12",
    "hmmt": "2025-02-15",
    "smt": "2025-05-05",
    "brumo": "2025-04-08",
}

if not args.cost and len(args.comps) == 1:
    ax.axvline(x=datetime.strptime(comp_dates[args.comps[0]], "%Y-%m-%d"), color='black', linestyle='--', linewidth=1)
    # Format x-axis for dates with month abbreviations and year
if not args.cost:
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # Show ticks every month
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))  # Format as 'Jan 25', 'Feb 25', etc.
# ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1)) # Show ticks every month
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.xticks(rotation=45, ha='right', fontsize=12)  # Increased font size for x-axis tick labels
# Increase font size for y-axis tick labels
plt.yticks(fontsize=12)

# Set labels and title
if not args.cost:
    ax.set_xlabel("Model Release Date", fontsize=14)
else:
    ax.set_xlabel("Model Cost", fontsize=14)
    ax.set_xscale('log')
ax.set_ylabel("Model Score", fontsize=14)
sns.despine(left=True, bottom=True)

    
# set background color to grey
ax.set_facecolor((0.97,0.97,0.97))
# if comp == "aime":
#     ax.set_title("Model Performance vs Release Date (AIME 2025)", fontsize=16)
# else:
#     ax.set_title("Model Performance vs Release Date (HMMT 2025)", fontsize=16)
ax.grid(True)

plt.tight_layout()
plt.savefig(args.save_file)
plt.show()




