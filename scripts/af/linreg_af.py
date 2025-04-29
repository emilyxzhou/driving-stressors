import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import shap
import sys
import yaml
sys.path.append("/home/emilyzho/distracted-driving/src/")

from matplotlib.offsetbox import AnchoredText
from munch import munchify
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
import seaborn as sns

from tqdm import tqdm

from constants_af import *
from data_reader_af import load_dataset


PLOT_FOLDER = "/home/emilyzho/distracted-driving/figs/linreg"

sns.set_theme(palette="dark")
font = {"size": 20}
matplotlib.rc("font", **font)
plt.rcParams["axes.labelsize"] = 14

# Load config file
os.makedirs("ckpt", exist_ok=True)
with open("/home/emilyzho/distracted-driving/config.yaml", "r") as f:
    cfg = munchify(yaml.safe_load(f))


# Load data
# subjects, features, labels = load_dataset(normalized=False, apply_normalization=True, resample=False)
subjects, features, labels = load_dataset(normalized=True, apply_normalization=False, resample=False)

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
    model = LinearRegression()

    model.fit(X_train, y_train)

    # Evaluate performance
    r2 = model.score(X_test, y_test)
    print(f"R2: {r2}")
