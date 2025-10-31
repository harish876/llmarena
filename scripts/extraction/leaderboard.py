from comparison import get_intersection_configs, get_scores, quantile_from_pdf
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import scipy.stats as st
from human_score_data import human_scores
import yaml
from datetime import datetime
import os
np.random.seed(42)

def get_latex_model_name(model):
    if model == "o4-mini (high)":
        return r"\ofour~\textsc{(high)}"
    elif model == "o4-mini (medium)":
        return r"\ofour~\textsc{(medium)}"
    elif model == "o4-mini (low)":
        return r"\ofour~\textsc{(low)}"
    elif model == "geminipro":
        return r"\geminipro"
    elif model == "o3 (high)":
        return r"\othree~\textsc{(high)}"
    elif model == "Grok 3 Mini (high)":
        return r"\grokthreemini~\textsc{(high)}"
    elif model == "Grok 3 Mini (low)":
        return r"\grokthreemini~\textsc{(low)}"
    elif model == "Qwen3-235B-A22B":
        return r"\qwenthreebig"
    elif model == "Qwen3-30B-A3B":
        return r"\qwenthreesmall"        
    return r"\textsc{" + model + "}"

def permutation_test(samples, iterations=10000):
    observed_T = np.mean([np.mean(comp_s) for comp_s in samples])

    values = []
    n_samples = sum([len(comp_s) for comp_s in samples])
    random_integers = np.random.randint(0, 2, size=(iterations, n_samples)) * 2 - 1

    for iter in range(iterations):
        random_integers_here = random_integers[iter]
        samples_here = []
        for comp_s in samples:
            samples_here.append([sample_here * random_integers_here[i] for i, sample_here in enumerate(comp_s)])
            random_integers_here = random_integers_here[len(comp_s):]
        values.append(np.mean([np.mean(comp_s) for comp_s in samples_here]))
    
    values = np.array(values)

    return np.sum(values <= observed_T) / len(values)


def rank_variance(scores, alpha=0.05, n_reps=1000):
    rank_quantiles = {
        config_path: [] for config_path in scores
    }

    for config_path in tqdm(scores):
        n_below = 0
        n_up = 0
        for second_config_path in scores:
            if second_config_path == config_path:
                continue
            samples = []
            for comp_index in range(len(scores[config_path])):
                samples.append([])
                for question_index in range(len(scores[config_path][comp_index])):
                    samples[-1].append(float(scores[config_path][comp_index][question_index]) - float(scores[second_config_path][comp_index][question_index]))

            hat_P = permutation_test(samples, iterations=n_reps)
            if hat_P < alpha:
                n_below += 1
            hat_P_inverse = permutation_test([[-s for s in comp_s] for comp_s in samples], iterations=n_reps)
            if hat_P_inverse < alpha:
                n_up += 1
        rank_quantiles[config_path].append(n_below + 1)
        rank_quantiles[config_path].append(len(scores) - n_up)
    return rank_quantiles

def variance_of_overall_mean(score_model):
    """
    score_model : array-like shaped (n_runs, n_questions)
                  containing 0/1 accuracy outcomes
    returns var_mean, se_mean
    """
    scores = np.asarray(score_model)
    N = scores.size                      # total number of 0/1 observations
    p_hat = scores.mean()                # grand mean accuracy
    var_mean = p_hat * (1 - p_hat) / N   # same as scores.var(ddof=1) / N
    return var_mean  # CI half-width

def deviation(scores, alpha=.05):
    deviations = dict()
    for config_path in scores:
        variances_each_comp = [variance_of_overall_mean(comp_s) for comp_s in scores[config_path]]
        variance = (1 / len(scores[config_path]) ** 2) * np.sum(variances_each_comp)
        std = np.sqrt(variance) * st.norm.ppf(1 - alpha / 2)
        deviations[config_path] = std
    return deviations

def get_human_score_quantiles(comps, quantiles):
    if not all([comp in human_scores.keys() for comp in comps]):
        return None
    
    scores = [
        np.mean([quantile_from_pdf(human_scores[comp], quantile, "aime" in comp) for comp in comps])
        for quantile in quantiles
    ]
    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--comps", type=str, nargs="+", default=["aime/aime_2025", "hmmt/hmmt_feb_2025"])
    parser.add_argument("--output-folder", type=str, default="outputs")
    parser.add_argument("--configs-folder", type=str, default="configs/models")
    parser.add_argument("--competition-config-folder", type=str, default="configs/competitions")
    parser.add_argument("--save-file", type=str, default="rank_variance.jsonl")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-repetitions", type=int, default=5000)
    parser.add_argument("--keep-comps", action="store_true", help="Keep the competition names in the output")
    parser.add_argument("--compute-variance", action="store_true", help="Compute variance")
    parser.add_argument("--only-intersection", action="store_true", help="Only keep the intersection of configs")
    parser.add_argument("--no-cost", action="store_true", help="Do not compute cost")
    parser.add_argument("--human-quantiles", type=float, nargs="+", default=[])
    parser.add_argument("--output-format", type=str, choices=["table", "latex"], default="table",
                        help="Format for final printed results")
    parser.add_argument("--exclude-models", type=str, nargs="+", default=[],
                        help="Exclude configs whose YAML filename contains any of these substrings")

    args = parser.parse_args()

    if args.compute_variance:
        args.only_intersection = True
    
    all_existing_configs = get_intersection_configs(args.comps, args.output_folder,
                                                    args.configs_folder, args.competition_config_folder, 
                                                    intersection=args.only_intersection)
    # Optionally exclude configs by substring match on YAML filename (basename without path)
    if args.exclude_models:
        def should_exclude(config_path: str) -> bool:
            base = os.path.basename(config_path).lower()  # e.g., "gpt-5.yaml" -> from path part
            # config_path currently excludes extension in usage; match on this string regardless
            return any(exclude.lower() in base or exclude.lower() in config_path.lower()
                       for exclude in args.exclude_models)

        all_existing_configs = {
            config_path: name for config_path, name in all_existing_configs.items()
            if not should_exclude(config_path)
        }
    competition_dates = {
        comp: yaml.safe_load(open(f"{args.competition_config_folder}/{comp}.yaml", "r"))["date"] for comp in args.comps
    }
    competition_dates = {comp: datetime.strptime(date, "%Y-%m-%d") if isinstance(date, str) else datetime.strptime("2000-01-01", "%Y-%m-%d") for comp, date in competition_dates.items()}
    
    if args.compute_variance:
        scores, _ = get_scores(args.comps, args.output_folder,
                            args.competition_config_folder, all_existing_configs, avg=False)
        all_scores = dict()
        for config_path in scores:
            all_scores[config_path] = []
            for comp in args.comps:
                all_scores[config_path].append(np.array(scores[config_path][comp]).astype(float).reshape(-1))

        max_length = max([sum([len(score_comp) for score_comp in all_scores[config_path]]) for config_path in all_scores])
        # drop all configs that dont have max_length
        all_scores = {
            config_path: all_scores[config_path] for config_path in all_scores if sum([len(score_comp) for score_comp in all_scores[config_path]]) == max_length
        }
        all_existing_configs = {
            config_path: all_existing_configs[config_path] for config_path in all_existing_configs if config_path in all_scores
        }
        rank_quantiles = rank_variance(all_scores, alpha=args.alpha, n_reps=args.n_repetitions)

        deviations = deviation(all_scores, alpha=args.alpha)
    else:
        deviations = dict()
        rank_quantiles = dict()
        for config_path in all_existing_configs:
            deviations[config_path] = 0
            rank_quantiles[config_path] = [0, 0]

    avg_scores, avg_cost = get_scores(args.comps, args.output_folder,
                            args.competition_config_folder, all_existing_configs, avg=True)

    final_df = []

    for config_path in avg_scores:
        model_date = yaml.safe_load(open(f"{args.configs_folder}/{config_path}.yaml", "r")).get("date", "2000-01-01")
        model_date = datetime.strptime(model_date, "%Y-%m-%d")
        extra = dict()
        if args.keep_comps:
            for comp in args.comps:
                cellcolor = ""
                if model_date < competition_dates[comp] and avg_scores[config_path][comp] != "N/A":
                    cellcolor = r"\cellcolor{LightGreen}"
                if avg_scores[config_path][comp] != "N/A":
                    avg_scores[config_path][comp] *= 100
                else:
                    avg_scores[config_path][comp] = "{N/A}"
                extra[comp.split("/")[1].split("_")[0].upper()] = cellcolor + str(avg_scores[config_path][comp])
        avg_cost_model = np.mean([score for score in avg_cost[config_path].values() if score != "N/A"])
        if avg_cost_model < 5 * 1e-3:
            if args.output_format == "latex":
                avg_cost_model = "{$<10^{-2}$}"
            else:
                avg_cost_model = avg_cost_model
                
        avg_score = np.mean([score for score in avg_scores[config_path].values() if score not in ["{N/A}", "N/A"]])
        if not args.keep_comps:
            avg_score *= 100
        deviation_string = ""
        start_score = ""
        if args.compute_variance:
            start_score = "$"
            deviation_string = f"\\pm {deviations[config_path] * 100:.1f}$"
        model_display_name = (
            get_latex_model_name(all_existing_configs[config_path])
            if args.output_format == "latex"
            else all_existing_configs[config_path]
        )
        final_df.append({
            "Model": model_display_name,
            **extra,
            "Rank": f"{rank_quantiles[config_path][0]}-{rank_quantiles[config_path][1]}",
            "Acc (avg)": f"{start_score} {avg_score:.2f} {deviation_string}",
            "acc": avg_score,
            "Cost (avg)": avg_cost_model if not any(x in config_path for x in ["thinking-2.0", "other/"]) else "{N/A}",
        })
        if args.no_cost:
            final_df[-1].pop("Cost (avg)")
        if not args.compute_variance:
            final_df[-1].pop("Rank")
    
    human_quantiles = get_human_score_quantiles(args.comps, [1 - quant for quant in args.human_quantiles])
    for i, quantile in enumerate(args.human_quantiles):
        quantile_round = round(quantile * 100, 1)
        if int(quantile_round) == quantile_round:
            quantile_round = int(quantile_round)
        final_df.append({
            "Model": f"human (top {quantile_round}%)",
            "acc": human_quantiles[i] * 100,
            "Acc (avg)": human_quantiles[i] * 100,
        })

    final_df = pd.DataFrame(final_df)
    # sort by acc
    final_df = final_df.sort_values(by="acc", ascending=False)
    # reset index to be 1-based after sorting
    final_df.index = np.arange(1, len(final_df) + 1)
    final_df.to_json(args.save_file, orient="records", lines=True)
    del final_df["acc"]
    # print final results in requested format
    if args.output_format == "latex":
        df_latex = final_df.copy()
        # all column names -> {\textbf{name}}
        df_latex.columns = [f"{{\\textbf{{{col}}}}}" for col in df_latex.columns]
        print(df_latex.to_latex(index=False))
    else:
        # pretty console table
        pd.set_option("display.max_colwidth", None)
        pd.set_option("display.width", 200)
        try:
            from tabulate import tabulate
            # show 1-based index in the console table
            print(tabulate(final_df, headers="keys", tablefmt="grid", showindex=True))
        except Exception:
            # Fallback if tabulate is unavailable
            print(final_df.to_string(index=True))
