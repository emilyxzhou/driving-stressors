import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.constants_trina import *

RESULTS_PATH = "/home/emilyzho/distracted-driving/results/results_xgb_trina10"
PLOT_PATH = "/home/emilyzho/distracted-driving/figs/predictions"


def plot_proba_ts(sub, y_pred, bounds, offset_list, event_list):
    y_pred = np.array(y_pred)
    y_pred_t = np.mean(y_pred, axis=0)

    plt.figure(figsize=(15, 3))
    plt.plot(y_pred_t, color="navy", linewidth=0.99, alpha=0.75)

    for xc in offset_list:
        plt.axvline(x=xc, color="crimson")

    plt.title(f"{sub} - {event_list}")
    plt.ylabel("Anomaly probability")
    plt.xlabel("Time (s)")
    plt.xticks(np.arange(0, len(y_pred_t), 30), np.arange(0, len(y_pred_t), 30))
    plt.grid(linestyle="--", alpha=0.5, linewidth=0.75)
    plt.xlim(-5, bounds[1] + 5)
    plt.ylim(-0.05, 1.05)

    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/{sub}.png", bbox_inches="tight", dpi=300)
    plt.close()

    # Save the data
    to_json = {
        "y_pred": y_pred.tolist(),
        "offset_list": offset_list.tolist(),
        "event_names": event_list,
    }
    os.makedirs("ts_data", exist_ok=True)
    with open(f"ts_data/{sub}.json", "w") as f:
        json.dump(to_json, f, indent=4)


def overlap_add(pred_ts, win=15):
    """
    Applies overlap-add smoothing to a time series of scalar predictions.
    """
    total_length = len(pred_ts)
    stress_sum = np.zeros(total_length)
    count = np.zeros(total_length)

    for i, pred in enumerate(pred_ts):
        start = max(0, i - win // 2)
        end = min(total_length, i + win // 2 + (win % 2))
        stress_sum[start:end] += pred
        count[start:end] += 1

    return stress_sum / count


def get_scores(path0, path1):
    pred_0 = np.load(path0)
    pred_1 = np.load(path1)

    y_pred = np.concatenate([pred_0, pred_1])
    y_true = np.concatenate([np.zeros(len(pred_0)), np.ones(len(pred_1))])

    return (
        {
            "ROC-AUC": roc_auc_score(y_true, y_pred),
            "Proba 0": np.mean(pred_0),
            "Proba 1": np.mean(pred_1),
        },
        overlap_add(y_pred),
        # y_pred,
        y_true,
    )


if __name__ == "__main__":
    # LOSO, within trina10
    for subject in SUBJECTS[0:1]:
        path0 = f"{RESULTS_PATH}/{subject}/0.npy"
        path1 = f"{RESULTS_PATH}/{subject}/1.npy"

        plot_path = os.path.join(PLOT_PATH, f"trina_loso_{subject}.jpg")