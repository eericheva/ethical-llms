import glob

import matplotlib
import numpy as np
import pandas as pd
from natsort import natsorted

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from UDHR.get_UDHR_inputs import get_identities_dicts, get_rights_lists

color_names = [
    "dusty purple",
    "orange",
    "dark tan",
    "pink",
    "baby blue",
    "olive",
    "sea blue",
    "dusty red",
    "faded green",
    "amber",
    "windows blue",
]
colors = sns.xkcd_palette(color_names)

i_names = np.array(list(get_identities_dicts().keys()))
r_names = np.array([r for r in get_rights_lists()[0]])


def get_results_list():
    results_mfq = natsorted(glob.glob("UDHR/responses/*.csv", recursive=True))
    results_mfq = [fn for fn in results_mfq if "_l_" in fn]
    scores_before = np.zeros((1, len(r_names), len(i_names)))
    scores_after = np.zeros((len(results_mfq), len(r_names), len(i_names)))

    for ll, file_n in enumerate(results_mfq[:3]):
        print(f"Layer {ll}")
        iter = int(file_n.split("_rep_")[-1].split(".csv")[0])
        dat = pd.read_csv(file_n, delimiter="\t")
        dat["completions"] = np.clip(dat["completions"], 0, 1)
        for ii, i_name in enumerate(i_names):
            for rr, r_name in enumerate(r_names):
                scores_before[0, rr, ii] = np.mean(
                    dat[
                        (dat.is_modified == False)
                        & (dat.right == r_name)
                        & (dat.id_class == i_name)
                    ]["completions"].values.astype(int)
                )
                scores_after[ll, rr, ii] = np.mean(
                    dat[
                        (dat.is_modified == True)
                        & (dat.right == r_name)
                        & (dat.id_class == i_name)
                    ]["completions"].values.astype(int)
                )
    return scores_before, scores_after


def get_pic(scores_before, scores_after):
    # scores = np.concatenate([scores_before, scores_after], dim = 0)
    fig, axes = plt.subplots(figsize=(4.5, 50.5))
    bar_height = 0.1
    offset = 0.2

    for ii, i_name in enumerate(i_names):
        axes.barh(
            y=r_names,
            width=list(scores_before[0, :, ii]),
            color="black",
            edgecolor="black",
            alpha=0.2,
            height=bar_height,
            label="Unsteered",
        )
        for ll in range(len(scores_after)):
            axes.barh(
                y=ll * bar_height,
                width=list(scores_after[ll, :, ii]),
                color=colors[ll + 1],
                edgecolor=colors[ll + 1],
                alpha=0.2,
                height=bar_height,
                label=f"Layer {ll}",
            )
            a = 0
    plt.savefig("UDHR/responses/ff.png")
    plt.show()


if __name__ == "__main__":
    scores_before, scores_after = get_results_list()
    get_pic(scores_before, scores_after)
