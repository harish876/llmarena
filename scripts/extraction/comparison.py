import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from matharena.configs import extract_existing_configs
import yaml
import json
import os
from loguru import logger
import numpy as np
import math
from matplotlib import rcParams
from human_score_data import human_scores
from datetime import datetime

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['text.usetex'] = False

def get_intersection_configs(comps, output_folder, configs_folder, competition_config_folder, intersection=True, model_filter=None, human_readable_id_overrides=None):
    configs = {
        comp: extract_existing_configs(comp, output_folder, configs_folder, 
                                       competition_config_folder,
                                       model_filter=model_filter,
                                       human_readable_id_overrides=human_readable_id_overrides)[1]
        for comp in comps
    }

    all_existing_configs = set(configs[comps[0]].keys())
    for comp, config in configs.items():
        if intersection:
            all_existing_configs = all_existing_configs.intersection(set(config.keys()))
        else:
            all_existing_configs = all_existing_configs.union(set(config.keys()))
    
    all_existing_configs = list(all_existing_configs)
    all_existing_configs = {
        config_path: configs[comps[0]][config_path]
        for config_path in all_existing_configs
    }
    return all_existing_configs

def quantile_from_pdf(pdf, quantile, is_aime=False):
    if not (0 <= quantile <= 1):
        raise ValueError("Quantile must be between 0 and 1")

    # Normalize PDF
    total = sum(pdf["scores"].values())
    score_diff = 0 if not is_aime else -1 # accidentally added + 1 when extracting
    max_key = pdf["max_score"]
    normalized_pdf = {k: v / total for k, v in sorted(pdf["scores"].items())}

    # Build CDF
    cdf = []
    cumulative = 0
    normalized_pdf = dict(sorted(normalized_pdf.items()))
    for score, prob in normalized_pdf.items():
        cumulative += prob
        cdf.append((score, cumulative))

    # Handle exact matches or boundaries
    if quantile <= cdf[0][1]:
        return (cdf[0][0] + score_diff) / max_key
    if quantile >= cdf[-1][1]:
        return (cdf[-1][0] + score_diff) / max_key

    # Find where the quantile fits and interpolate
    for i in range(1, len(cdf)):
        lower_score, lower_cum = cdf[i - 1]
        upper_score, upper_cum = cdf[i]
        if lower_cum <= quantile <= upper_cum:
            # Linear interpolation
            t = (quantile - lower_cum) / (upper_cum - lower_cum)
            return (score_diff + lower_score + t * (upper_score - lower_score)) / max_key

    raise RuntimeError("Quantile interpolation failed")  # Should never happen

def get_human_scores(old_comps, new_comps):
    if not all([comp in human_scores.keys() for comp in old_comps + new_comps]):
        return None, None
    
    quantiles = [i / 100 for i in range(101)]

    old_scores = [
        np.mean([quantile_from_pdf(human_scores[comp], quantile, "aime" in comp) for comp in old_comps ])
        for quantile in quantiles
        
    ]

    new_scores = [
        np.mean([quantile_from_pdf(human_scores[comp], quantile, "aime" in comp) for comp in new_comps])
        for quantile in quantiles
    ]
    return old_scores, new_scores

def compute_pass_at_k(correct_list, k):
    """
    Compute pass@k metric: probability that at least one of k sampled runs is correct.
    
    Formula: pass@k = 1 - C(N - c, k) / C(N, k)
    where N is total runs and c is number of correct runs.
    
    Args:
        correct_list: List of boolean values indicating correctness of each run
        k: Number of runs to sample
    
    Returns:
        float: pass@k score, or None if cannot compute
    """
    if not correct_list:
        return None
    
    # Convert to booleans if needed (handle mixed types)
    if any(isinstance(c, str) for c in correct_list):
        return None
    
    N = len(correct_list)
    correct_count = sum(correct_list)
    incorrect_count = N - correct_count
    
    # If we have fewer than k runs, pass@k is 1 if any run is correct, else 0
    if N < k:
        return 1.0 if correct_count > 0 else 0.0
    
    # If all runs are correct, pass@k = 1
    if correct_count == N:
        return 1.0
    
    # If all runs are incorrect, pass@k = 0
    if correct_count == 0:
        return 0.0
    
    # If we can't sample k incorrect runs, pass@k = 1
    if incorrect_count < k:
        return 1.0
    
    # Standard pass@k formula: 1 - C(N - c, k) / C(N, k)
    try:
        combinations_incorrect = math.comb(incorrect_count, k)
        combinations_total = math.comb(N, k)
        return 1.0 - (combinations_incorrect / combinations_total)
    except (ValueError, OverflowError):
        # Fallback for edge cases
        return None

def get_scores(comps, output_folder, competition_config_folder, all_existing_configs, 
                avg=True, check_complete=False, runs_per_problem=4, compute_pass_at_4=False):
    scores = dict()
    costs = dict()
    complete_flags = dict()
    pass_at_4_scores = dict() if compute_pass_at_4 else None

    for comp in comps:
        with open(f"{competition_config_folder}/{comp}.yaml", "r") as f:
            competition_config = yaml.safe_load(f)
        n_problems = competition_config["n_problems"]
        
        for config_path in all_existing_configs:
            results = []
            costs_here = []
            pass_at_4_results = [] if compute_pass_at_4 else None
            is_complete = True
            for i in range(1, n_problems + 1):
                if not os.path.exists(f"{output_folder}/{comp}/{config_path}/{i}.json"):
                    logger.warning(f"File {output_folder}/{comp}/{config_path}/{i}.json does not exist")
                    is_complete = False
                    break
                results_prob = json.load(open(f"{output_folder}/{comp}/{config_path}/{i}.json", "r"))
                if check_complete and results_prob.get("N", 0) < runs_per_problem:
                    is_complete = False
                correct = results_prob["correct"] if "usamo" not in comp else [
                    judgment["points"] / 7 for judgment in results_prob["judgment"][0]
                ]
                # randomly perturb correct
                # correct = [correct[idx] for idx in np.random.permutation(len(correct))]
                costs_sample = [c["cost"] for c in results_prob["detailed_costs"]]
                if avg:
                    results.extend(correct)
                    costs_here.extend([np.mean(costs_sample)])
                    if compute_pass_at_4:
                        pass_at_4 = compute_pass_at_k(correct, 4)
                        if pass_at_4 is not None:
                            pass_at_4_results.append(pass_at_4)
                else:
                    results.append(correct)
                    costs_here.append(np.mean(costs_sample))
                    if compute_pass_at_4:
                        pass_at_4 = compute_pass_at_k(correct, 4)
                        if pass_at_4_results is None:
                            pass_at_4_results = []
                        pass_at_4_results.append(pass_at_4 if pass_at_4 is not None else [])
            scores[config_path] = scores.get(config_path, dict())
            costs[config_path] = costs.get(config_path, dict())
            if compute_pass_at_4:
                pass_at_4_scores[config_path] = pass_at_4_scores.get(config_path, dict())
            if check_complete:
                complete_flags[config_path] = complete_flags.get(config_path, dict())
                complete_flags[config_path][comp] = is_complete
            if len(results) == 0:
                scores[config_path][comp] = "N/A"
                costs[config_path][comp] = "N/A"
                if compute_pass_at_4:
                    pass_at_4_scores[config_path][comp] = "N/A"
            elif avg:
                scores[config_path][comp] = sum(results) / len(results)
                costs[config_path][comp] = sum(costs_here)
                if compute_pass_at_4 and pass_at_4_results:
                    pass_at_4_scores[config_path][comp] = np.mean(pass_at_4_results)
                elif compute_pass_at_4:
                    pass_at_4_scores[config_path][comp] = "N/A"
            else:
                scores[config_path][comp] = results
                costs[config_path][comp] = costs_here
                if compute_pass_at_4:
                    pass_at_4_scores[config_path][comp] = pass_at_4_results if pass_at_4_results else []
    if check_complete:
        if compute_pass_at_4:
            return scores, costs, complete_flags, pass_at_4_scores
        return scores, costs, complete_flags
    if compute_pass_at_4:
        return scores, costs, pass_at_4_scores
    return scores, costs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-comps", type=str, nargs="+", default=["aime/aime_2025_I", "aime/aime_2025_II", "hmmt/hmmt_feb_2025"])
    parser.add_argument("--old-comps", type=str, nargs="+", default=["aime/aime_2024_I", "aime/aime_2024_II", "hmmt/hmmt_feb_2024"])
    parser.add_argument("--output-folder", type=str, default="outputs")
    parser.add_argument("--configs-folder", type=str, default="configs/models")
    parser.add_argument("--competition-config-folder", type=str, default="configs/competitions")
    parser.add_argument("--save-file", type=str, default="paper_data/rank_correlation.pdf")
    args = parser.parse_args()

    all_existing_configs = get_intersection_configs(args.old_comps + args.new_comps, args.output_folder,
                                                    args.configs_folder, args.competition_config_folder)

    scores, _ = get_scores(args.old_comps + args.new_comps, args.output_folder, 
                        args.competition_config_folder, all_existing_configs)
    
    human_old_scores, human_new_scores = get_human_scores(args.old_comps, args.new_comps)

    new_configs_date = [
        yaml.safe_load(open(f"{args.competition_config_folder}/{comp}.yaml", "r"))["date"] for comp in args.new_comps
    ]
    new_configs_date = [datetime.strptime(date, "%Y-%m-%d") for date in new_configs_date]
    oldest_new_config_date = max(new_configs_date)
    # print as human readable date, with full month name

    model_released_dates = {
        config_path: datetime.strptime(yaml.safe_load(open(f"{args.configs_folder}/{config_path}.yaml", "r")).get("date", "2000-01-01"), "%Y-%m-%d")
        for config_path in all_existing_configs
    }

    model_released_before = {
        config_path: "released before" if model_released_dates[config_path] < oldest_new_config_date else "released after"
        for config_path in all_existing_configs
    }

    for config_path in scores:
        scores[config_path]["old"] = np.mean([scores[config_path][comp] for comp in args.old_comps])
        scores[config_path]["new"] = np.mean([scores[config_path][comp] for comp in args.new_comps])
    
    # plot
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.despine(left=True, bottom=True)
    ax.set_facecolor((0.97, 0.97, 0.97))

    # do scatterplot of old vs new

    before_color = "#1f77b4"
    after_color = "#ff7f0e"
    for config_path in scores:
        ax.scatter(scores[config_path]["new"], scores[config_path]["old"], 
                   s=100, color=before_color if model_released_before[config_path] == "released before" else after_color)
    # make legend with the two colors
    if "hmmt" in args.new_comps[0]:
        name = "HMMT"
    else:
        name = "AIME"
    ax.scatter([], [], s=100, color=before_color, label=f"Released before {name} 2025")
    ax.scatter([], [], s=100, color=after_color, label=f"Released after {name} 2025")
    ax.set_xlabel(f"{name} 2025")
    ax.set_ylabel(f"{name} 2024")
    # set size of the axis labels
    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)
    # set size of the ticks
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    if human_old_scores is None:
        ax.plot([0, 1], [0, 1], color="black", linestyle="--", label="Identity")
    else:
        ax.plot(human_new_scores, human_old_scores, color="black", linestyle="--", label="Human")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # set legend font size
    ax.legend(fontsize=13, loc="lower right")
    plt.tight_layout()
    plt.savefig(args.save_file, bbox_inches='tight')
    plt.show()
