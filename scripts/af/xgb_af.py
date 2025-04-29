import numpy as np
import os
import sys
import yaml
sys.path.append("/home/emilyzho/distracted-driving/src/")

from munch import munchify
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score
from sklearn.impute import KNNImputer
from xgboost import XGBClassifier

from tqdm import tqdm

from constants_af import *
from data_reader_af import load_dataset


# Load config file
os.makedirs("ckpt", exist_ok=True)
with open("/home/emilyzho/distracted-driving/config.yaml", "r") as f:
    cfg = munchify(yaml.safe_load(f))


# Load data
# subjects, features, labels = load_dataset(normalized=False, binarize_threshold="median", apply_normalization=True, resample=False)
# subjects, features, labels = load_dataset(normalized=False, binarize_threshold="median", apply_normalization=False, resample=False)
subjects, features, labels = load_dataset(normalized=True, scaled=False, binarize_threshold="median", apply_normalization=False, resample=False)
# subjects, features, labels = load_dataset(normalized=True, scaled=True, binarize_threshold="median", apply_normalization=False, resample=False)

# Drop SCR ratio feature column
features = features.drop("scr_ratio", axis=1)

subjects = subjects.to_numpy()
features = features.to_numpy()
labels = labels.to_numpy()

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

    # imp = KNNImputer(n_neighbors=3)
    # X_train = np.where(X_train == 2048.0, np.nan, X_train)
    # X_train = imp.fit_transform(X_train)

    # X_test = np.where(X_test == 2048.0, np.nan, X_test)
    # X_test = imp.fit_transform(X_test)

    # Load the model
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=cfg.seed,
        reg_lambda=10
    )

    model.fit(X_train, y_train)

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

    folder = f"results_xgb_af"
    os.makedirs(folder, exist_ok=True)
    np.save(f"{folder}/{test_sub}_0.npy", probas_0)
    np.save(f"{folder}/{test_sub}_1.npy", probas_1)