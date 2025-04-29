import matplotlib.pyplot as plt
import numpy as np
import os
import shap
import sys
import yaml
sys.path.append("/home/emilyzho/distracted-driving/src/")

from munch import munchify
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from tqdm import tqdm

from constants_dd import *
from data_reader_dd import load_dataset


TOYOTA_SUBJECTS = [f"P{i}" for i in range(1, 34) if i != 3]
INCLUDE_VIDEO = False

every = 1
scale = True
feature_ids = [0, 2, 3, 4, 5, 6, 7, 8]


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

X_train = np.concatenate(X_train, axis=0)[:, feature_ids]
y_train = np.concatenate(y_train, axis=0)

print(X_train.shape)

# Fill in missing values
imp = KNNImputer(n_neighbors=3)
X_train = imp.fit_transform(X_train)

# Drop extra features not in distracted-driving
X_train = X_train[:, [0, 2, 3, 4, 5, 6, 7]]

# Load test data 
subjects, X_test, y_test = load_dataset()
# Drop SCR ratio feature column
X_test = np.delete(X_test, 5, axis=1)


# Rearrange feature columns to match Toyota
new_order = [1, 2, 3, 4, 5, 6, 0]
X_test = X_test[:, new_order]

# Fill in missing values
X_test = imp.fit_transform(X_test)

# Check for label imbalance
test_zeros = np.count_nonzero(y_test == 0)
test_ones = np.count_nonzero(y_test == 1)
print(test_zeros)
print(test_ones)

# Load the model
model = XGBClassifier(
    eval_metric="logloss", reg_lambda=5, random_state=cfg.seed, subsample=0.8
)

model.fit(X_train, y_train)

scores = []

# Evaluate performance
y_proba = model.predict_proba(X_test)
y_pred = np.argmax(y_proba, axis=1)
y_proba = y_proba[:, 1]

auroc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {auroc:.4f}")
scores.append(auroc)

# Save the predictions
probas_0 = y_proba[np.where(y_test == 0)[0]]
probas_1 = y_proba[np.where(y_test == 1)[0]]

folder = f"results_xgb_toyota_pretrain{cfg.seed}"
os.makedirs(folder, exist_ok=True)
np.save(f"{folder}/0.npy", probas_0)
np.save(f"{folder}/1.npy", probas_1)

feature_names = [
    # ECG
    "HR",
    # EDA
    "SCL mean",
    "SCL slope",
    "SCR count",
    "SCR amp",
    "SCR rise",
    # RSP
    "RSP rate"
]
explainer = shap.Explainer(model, feature_names=feature_names)
shap_values = explainer(X_test)
shap.plots.beeswarm(shap_values)
fig_path = "figs/toyota_pretrain_shap.png"
plt.savefig(fig_path, dpi=300, bbox_inches="tight")

print(model.feature_importances_)
