import pandas as pd
import json
import os
import glob
import numpy as np

output_folder = "outputs/imo/imo_2025"

results = dict()
costs = dict()

for file in glob.glob(os.path.join(output_folder, "**/*.json"), recursive=True):
    with open(file, "r") as f:
        data = json.load(f)
        problem_idx = os.path.basename(file).replace(".json", "")
        model_name = os.path.basename(os.path.dirname(file))
        if model_name not in results:
            results[model_name] = dict()
            costs[model_name] = 0
        results[model_name][problem_idx] = np.mean(data["correct"]) * 7
        costs[model_name] += sum([c["cost"] for c in data["detailed_costs"]]) / 4

# print as latex table, with model name in first column, problem score in each row, and then a final total, and then cost using pandas
df = pd.DataFrame(results).T
print(df)
df = df[[str(i) for i in range(1, 7)]]
df["Total"] = df.sum(axis=1)
df["Cost"] = df.index.map(lambda x: costs[x])
df = df.sort_values(by="Total", ascending=False)
print(df.to_latex(float_format="%.4f"))