from comparison import get_intersection_configs, get_scores
import argparse
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['text.usetex'] = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--comps", type=str, nargs="+", default=["aime/aime_2025", "hmmt/hmmt_feb_2025"])
    parser.add_argument("--output-folder", type=str, default="outputs")
    parser.add_argument("--configs-folder", type=str, default="configs/models")
    parser.add_argument("--competition-config-folder", type=str, default="configs/competitions")
    parser.add_argument("--save-file", type=str, default="paper_data/rank_correlation.pdf")
    args = parser.parse_args()

    all_existing_configs = get_intersection_configs(args.comps, args.output_folder,
                                                    args.configs_folder, args.competition_config_folder, True)
    scores, _ = get_scores(args.comps, args.output_folder,
                        args.competition_config_folder, all_existing_configs)
    
    comp_scores = {
        comp: [scores[config_path][comp] for config_path in scores] for comp in args.comps
    }

    correlation_matrix = np.zeros((len(args.comps), len(args.comps)))
    rank_correlation_matrix = np.zeros((len(args.comps), len(args.comps)))

    for i, comp1 in enumerate(args.comps):
        for j, comp2 in enumerate(args.comps):
            corr = np.corrcoef(comp_scores[comp1], comp_scores[comp2])[0, 1]
            correlation_matrix[i, j] = corr
            # rank correlation
            rank_corr, _ = spearmanr(comp_scores[comp1], comp_scores[comp2])
            rank_correlation_matrix[i, j] = rank_corr

    fig, ax = plt.subplots(figsize=(7, 7))

    comps_display = [comp.split("/")[-1].replace("_2025", "").replace("_feb", "") for comp in args.comps]

    # set font size larger
    plt.rcParams.update({'font.size': 16})
    sns.heatmap(rank_correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True,
                xticklabels=comps_display, yticklabels=comps_display, ax=ax, vmin=0, vmax=1)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', fontsize=18)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=18)
    ax.set_aspect('equal')
    plt.tight_layout()
    # remove the legend
    plt.gca().collections[0].colorbar.remove()

    plt.savefig(args.save_file, bbox_inches='tight')

    plt.show()
