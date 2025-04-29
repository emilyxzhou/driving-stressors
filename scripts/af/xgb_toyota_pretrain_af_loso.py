import matplotlib.pyplot as plt
import numpy as np
import os
import shap
import sys
import yaml
sys.path.append("/home/emilyzho/distracted-driving/src/")

from munch import munchify
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from tqdm import tqdm

from constants_af import *
from data_reader_af import load_dataset


TOYOTA_SUBJECTS = [f"P{i}" for i in range(1, 34) if i != 3]
INCLUDE_VIDEO = False

every = 1
scale = True
feature_ids = [0, 2, 3, 4, 5, 6]

# Load config file
os.makedirs("ckpt", exist_ok=True)
with open("/home/emilyzho/distracted-driving/config.yaml", "r") as f:
    cfg = munchify(yaml.safe_load(f))

# Load training data (Toyota)
X_train, y_train = [], []
for sub in TOYOTA_SUBJECTS:
    for task in cfg.tasks:
        task_file = f"{sub}_{task}_free.npy"
        if os.path.exists(cfg.token_root + task_file):
            baseline = np.load(cfg.token_root + task_file)
            if len(baseline) > 60:
                baseline = baseline[:60]
            mean, std = baseline.mean(axis=0), baseline.std(axis=0) + 1e-6

        task_file = f"{sub}_{task}_free.npy"
        if os.path.exists(cfg.token_root + task_file):
            this_X = np.load(cfg.token_root + task_file)[::every]
            this_X = (this_X - mean) / std if scale else this_X - mean
            X_train.append(this_X)
            y_train.append([0] * len(this_X))

        task_file = f"{sub}_{task}_event.npy"
        if os.path.exists(cfg.token_root + task_file):
            this_X = np.load(cfg.token_root + task_file)[::every]
            this_X = (this_X - mean) / std if scale else this_X - mean
            X_train.append(this_X)
            y_train.append([1] * len(this_X))

X_train = np.concatenate(X_train, axis=0)
y_train = np.concatenate(y_train, axis=0)

# Fill in missing values
imp = KNNImputer(n_neighbors=3)
X_train = imp.fit_transform(X_train)

# Drop extra features not in test dataset
X_train = X_train[:, feature_ids]

# Load test data 
# subjects, X_test, y_test = load_dataset(normalized=False, binarize_threshold="median", apply_normalization=True, resample=False)
# subjects, X_test, y_test = load_dataset(normalized=False, binarize_threshold="median", apply_normalization=False, resample=False)
# subjects, X_test, y_test = load_dataset(normalized=True, scaled=False, binarize_threshold="median", apply_normalization=False, resample=False)
subjects, X_test, y_test = load_dataset(normalized=True, scaled=False, binarize_threshold="median", apply_normalization=False, resample=False)
# Drop SCR ratio feature column
X_test = X_test.drop(columns=["scr_ratio", "scr_rise"], axis=1)

unique_subjects = np.unique(subjects)
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()

# Rearrange feature columns to match Toyota
new_order = [0, 2, 3, 4, 5, 1]
X_test = X_test[:, new_order]

# Check for label imbalance
# test_zeros = np.count_nonzero(y_test == 0)
# test_ones = np.count_nonzero(y_test == 1)

# Load the model
scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=cfg.seed,
    reg_lambda=10
)

# Set up CV
loso = LeaveOneOut()

# LOSO, training on trina_33 + all but one AF subject
for i, (train_index, test_index) in enumerate(loso.split(unique_subjects)):
    train_subs = np.array(unique_subjects)[train_index]
    test_sub = np.array(unique_subjects)[test_index][0]
    print(f"\nTest subject: {test_sub}")

    train_indices = np.where(np.isin(subjects, train_subs))
    test_indices = np.where(subjects==test_sub)
    # print(train_indices)
    # print(test_indices)

    # Split into training and test sets
    X_train_af = X_test[train_indices, :][0]
    y_train_af = y_test[train_indices]
    X_test_af = X_test[test_indices, :][0]
    y_test_af = y_test[test_indices]

    X_train_all = np.concatenate((X_train, X_train_af), axis=0)
    y_train_all = np.concatenate((y_train, y_train_af), axis=0)

    model.fit(X_train_all, y_train_all)

    scores = []

    # Evaluate performance
    test_zeros = np.count_nonzero(y_test_af == 0)
    test_ones = np.count_nonzero(y_test_af == 1)

    y_proba = model.predict_proba(X_test_af)
    y_pred = np.argmax(y_proba, axis=1)
    y_proba = y_proba[:, 1]

    auroc = roc_auc_score(y_test_af, y_proba)
    print(f"ROC-AUC: {auroc:.4f}")
    scores.append(auroc)

    y_pred = y_proba.copy()
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1
    acc = accuracy_score(y_test_af, y_pred)
    print(f"Acc: {acc:.4f}")
    scores.append(acc)

    # Save the predictions
    probas_0 = y_proba[np.where(y_test_af == 0)[0]]
    probas_1 = y_proba[np.where(y_test_af == 1)[0]]

    folder = f"results_xgb_toyota_pretrain_trina10_loso/{test_sub}"
    folder = os.path.join("/home/emilyzho/distracted-driving/results", folder)
    os.makedirs(folder, exist_ok=True)
    np.save(f"{folder}/0.npy", probas_0)
    np.save(f"{folder}/1.npy", probas_1)

# feature_names = [
#     # ECG
#     "HR",
#     # EDA
#     "SCL mean",
#     "SCL slope",
#     "SCR amp",
#     "SCR rise",
#     # RSP
#     "RSP period"
# ]
# explainer = shap.Explainer(model, feature_names=feature_names)
# shap_values = explainer(X_test)
# shap.plots.beeswarm(shap_values)
# fig_path = "/home/emilyzho/distracted-driving/figs/toyota_pretrain_af_shap.png"
# plt.savefig(fig_path, dpi=300, bbox_inches="tight")

# print(model.feature_importances_)
