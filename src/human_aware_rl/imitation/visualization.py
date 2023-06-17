import os
import re
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from human_aware_rl.utils import set_style

current_file_dir = os.path.dirname(os.path.abspath(__file__))
bc_dir = os.path.join(current_file_dir, "bc_runs")


def visualize(res_dir):
    algos_to_name = {
        "BC+H_proxy_0": "BC+H$_{Proxy}$",
        "BC+H_proxy_1": None,
        "BC+PPO_0": "BC+PPO",
        "BC+PPO_1": None,
        "BC+Scripted_0": "BC+Script",
        "BC+Scripted_1": None,
    }

    histogram_algos_5_layout = {
        "bc0": (
            ["BC+H_proxy_0",
             "BC+H_proxy_1",
             "BC+PPO_0",
             "BC+PPO_1",
             "BC+Scripted_0",
             "BC+Scripted_1"],
            None
        )
    }

    final_data = defaultdict(lambda: defaultdict(int))
    for layout in [
        "cramped_room",
        "coordination_ring",
        "asymmetric_advantages",
        "random0",  # forced coordination
        "random3",  # counter circuit
    ]:
        with open(os.path.join(res_dir, f"{layout}_raw.txt"), "r") as f:
            # TODO: parse shit
            pass

    mean_by_algo, std_by_algo = means_and_stds_by_algo(final_data)

    # Code from ipynb
    for hist_type, hist_algo_list in histogram_algos_5_layout.items():
        hist_algos, h_line_algo = hist_algo_list
        set_style()

        fig, ax0 = plt.subplots(1, figsize=(18, 6))  # figsize=(20,6))

        if hist_type not in ['humanai', 'humanai_base']:
            plt.rc('legend', fontsize=18)
            plt.rc('axes', titlesize=25)
        else:
            plt.rc('legend', fontsize=21)
            plt.rc('axes', titlesize=25)
        ax0.tick_params(axis='x', labelsize=18.5)
        ax0.tick_params(axis='y', labelsize=18.5)

        if hist_type != "cp":
            N = 5
        else:
            N = 3

        ind = np.arange(N)
        width = 0.1
        if hist_type not in ['humanai', 'humanai_base']:
            deltas = [-2.9, -1.5, -0.5, 0.5, 1.9, 2.9, 3.9]
        else:
            deltas = [-2.9, -1.5, -0.5, 0.5, 1.9, 2.9, 3.9]  # [-1, 0, 1, 2, 2.5, 3]

        for i in range(len(hist_algos)):
            delta, algo = deltas[i], hist_algos[i]
            color, hatch, alpha = get_algorithm_color(algo), get_texture(algo), get_alpha(algo)
            offset = ind + delta * width

            if algo == "PPO+BC_test":
                continue

            print(algo, mean_by_algo[algo])
            if algo in ["PPO_SP+PPO_SP", "PBT+PBT", "CP+CP"]:
                ax0.bar(offset, mean_by_algo[algo], width, color='none', edgecolor='gray', lw=1., zorder=0,
                        linestyle=':')
                ax0.bar(offset, mean_by_algo[algo], width, label=algos_to_name[algo], hatch="",
                        yerr=std_by_algo[algo], color='none', edgecolor='gray', linestyle=':')
            else:
                ax0.bar(offset, mean_by_algo[algo], width, label=algos_to_name[algo], yerr=std_by_algo[algo],
                        color=color, hatch=hatch, alpha=alpha)

        if h_line_algo is not None:
            for h_line in h_line_algo:
                ax0.hlines(final_data['cramped_room'][h_line][0], xmin=-0.4, xmax=0.4, colors="red",
                           label=algos_to_name[h_line], linestyle=':')
                ax0.hlines(final_data['asymmetric_advantages'][h_line][0], xmin=0.6, xmax=1.4, colors="red", linestyle=':')
                ax0.hlines(final_data['coordination_ring'][h_line][0], xmin=1.6, xmax=2.4, colors="red", linestyle=':')
                ax0.hlines(final_data['random0'][h_line][0], xmin=2.6, xmax=3.45, colors="red", linestyle=':')
                ax0.hlines(final_data['random3'][h_line][0], xmin=3.6, xmax=4.4, colors="red", linestyle=':')

        ax0.set_ylabel('Average reward per episode')
        ax0.set_title(graph_title(hist_type))

        ax0.set_xticks(ind + width / 3)
        ax0.set_xticklabels(('Cramped Rm.', 'Asymm. Adv.', 'Coord. Ring', 'Forced Coord.', 'Counter Circ.'))

        ax0.tick_params(axis='x', labelsize=18)

        if hist_type not in ['humanai', 'humanai_base']:
            handles, labels = ax0.get_legend_handles_labels()
            handles = switch_indices(0, 1, handles)
            labels = switch_indices(0, 1, labels)
            ax0.legend(handles, labels)

        # where some data has already been plotted to ax
        handles, labels = ax0.get_legend_handles_labels()

        # manually define a new patch
        patch = Patch(facecolor='white', edgecolor='black', hatch='/', alpha=0.5, label='Switched start indices')

        # handles is a list, so append manual patch
        handles.append(patch)

        # plot the legend
        ax0.legend(handles=handles, loc='best')

        ax0.set_ylim(0, 300)

        plt.savefig("data/images/" + hist_type + "_experiments.eps", format='eps', bbox_inches='tight')
        plt.show()


def get_algorithm_color(alg):
    opt_baseline_col = "#eaeaea"
    ours_col = '#fcd5b5'# orange #'#f79646'
    ours_other_col = "#F79646"
    other_col = '#4BACC6' # thiel
    other_other_col = "#2d6777"
    human_baseline_col = "#aeaeae"#"#c1c1c1"
    if alg == 'PBT':
        return other_col#opt_baseline_col
    elif alg == 'PPO_SP':
        return other_other_col #'#35ce47'#'#0000cc'
    elif alg == 'BC_Train':
        return ours_other_col #"#3884c9"
    elif alg == 'BC_Test':
        return ours_col #'#35ce47'#'#00cc00'
    else:
        raise ValueError(alg, "not recognized")


def get_texture(alg):
    return '/' if alg[-2:] == '_1' else ''


def get_alpha(alg):
    if alg == "avg_bc_test+bc_train":
        return 0.3
    else:
        return 1


def switch_indices(idx0, idx1, lst):
    lst = list(lst)
    lst[idx1], lst[idx0] = lst[idx0], lst[idx1]
    return lst


def graph_title(hist_type):
    return "Performance with human proxy model"


def means_and_stds_by_algo(full_data):
    mean_by_algo = defaultdict(list)
    std_by_algo = defaultdict(list)
    for layout, layout_algo_dict in full_data.items():
        for k in layout_algo_dict.keys():
            if type(layout_algo_dict[k]) is list or type(layout_algo_dict[k]) is tuple:
                mean, std = layout_algo_dict[k]
            else:
                mean, std = layout_algo_dict[k], 0
            mean_by_algo[k].append(mean)
            std_by_algo[k].append(std)
    return mean_by_algo, std_by_algo


if __name__ == "__main__":
    for pre in ["bc", "hproxy"]:
        for layout in [
            "random3",  # counter circuit
            "coordination_ring",
            "cramped_room",
            "random0",  # forced coordination
            "asymmetric_advantages",
        ]:
            for i in range(5):
                with open(os.path.join(bc_dir, "results", pre, f"{layout}_{i}.txt"), "r") as f:
                    s = f.read()
                    try:
                        spl = s.split("'ep_returns': array([")[1].split("])")[0].strip()
                    except:
                        continue
                    rs = [int(x) for x in re.split(r",\s*", spl)]
                    print(f"{pre} {layout.rjust(21, ' ')} {i}: mean: {np.mean(rs)}, se: {np.std(rs) / np.sqrt(len(rs))}")
