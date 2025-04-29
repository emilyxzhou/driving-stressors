import numpy as np
import os
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

from constants_trina import *
from data_reader_trina import load_dataset


# Load config file
os.makedirs("ckpt", exist_ok=True)
with open("/home/emilyzho/distracted-driving/config.yaml", "r") as f:
    cfg = munchify(yaml.safe_load(f))


# Load data
subjects, features, labels = load_dataset(normalized="z", resample=False, interpolate=True)
# subjects, features, labels = load_dataset(normalized="med", resample=False, interpolate=True)
features = features.drop(columns=["scr_ratio"])
features = features.to_numpy()
labels = labels.to_numpy()

# Drop SCR ratio feature column
features = np.delete(features, 5, axis=1)

# Set up CV
loso = LeaveOneOut()

scores = []

for i, (train_index, test_index) in enumerate(loso.split(SUBJECTS)):
    train_subs = np.array(SUBJECTS)[train_index]
    test_sub = np.array(SUBJECTS)[test_index][0]
    print(f"\nTest subject: {test_sub}")

    train_indices = np.where(np.isin(subjects, train_subs))
    test_indices = np.where(subjects==test_sub)

    # Split into training and test sets
    X_train = features[train_indices, :][0]
    y_train = labels[train_indices]
    X_test = features[test_indices, :][0]
    y_test = labels[test_indices]

    test_zeros = np.count_nonzero(y_test == 0)
    test_ones = np.count_nonzero(y_test == 1)
    # print(f"Positive: {test_ones}")
    # print(f"Negative: {test_zeros}")

    # imp = KNNImputer(n_neighbors=3)
    # X_train = np.where(X_train == 2048.0, np.nan, X_train)
    # X_train = imp.fit_transform(X_train)

    # X_test = np.where(X_test == 2048.0, np.nan, X_test)
    # X_test = imp.fit_transform(X_test)

    # Load the model
    model = XGBClassifier(
        eval_metric="logloss", reg_lambda=5, random_state=cfg.seed, subsample=0.8
    )

    model.fit(X_train, y_train)

    # Evaluate performance
    y_proba = model.predict_proba(X_test)
    y_pred = np.argmax(y_proba, axis=1)
    y_proba = y_proba[:, 1]

    auroc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC: {auroc:.4f}")
    scores.append(auroc)

    y_pred = y_proba.copy()
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1
    acc = accuracy_score(y_test, y_pred)
    print(f"Acc: {acc:.4f}")
    scores.append(acc)

    # Save the predictions
    probas_0 = y_proba[np.where(y_test == 0)[0]]
    probas_1 = y_proba[np.where(y_test == 1)[0]]

    folder = f"results_xgb_trina10/{test_sub}"
    folder = os.path.join("/home/emilyzho/distracted-driving/results", folder)
    os.makedirs(folder, exist_ok=True)
    np.save(f"{folder}/0.npy", probas_0)
    np.save(f"{folder}/1.npy", probas_1)