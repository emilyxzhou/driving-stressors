import csv
import glob
import numpy as np
import os
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.constants_af import *


def load_dataset(normalized=True, scaled=True, binarize_threshold=None, apply_normalization=False, resample=True):
    if normalized and not scaled:
        dataset_path = os.path.join(DATA_PROCESSED_FOLDER, "dataset_normalized.csv")
    elif normalized and scaled:
        dataset_path = os.path.join(DATA_PROCESSED_FOLDER, "dataset_norm_scaled.csv")
    else:
        dataset_path = os.path.join(DATA_PROCESSED_FOLDER, "dataset.csv")

    dataset = pd.read_csv(dataset_path, index_col=0)
    subjects = dataset.iloc[:, 0].unique().tolist()

    if not normalized and apply_normalization:
        # Perform subject-wise normalization
        scaler = StandardScaler()
        for subject in subjects:
            rows = dataset.loc[dataset["Subject"] == subject].index.tolist()
            dataset.iloc[rows, 1:-1] = scaler.fit_transform(dataset.iloc[rows, 1:-1])

    if binarize_threshold == "median":
        for subject in subjects:
            subject_indices = dataset[dataset["Subject"] == subject].index
            median_label = dataset[dataset["Subject"] == subject].loc[:, "label"].median()
            dataset.iloc[subject_indices, -1] = dataset.iloc[subject_indices, -1].apply(lambda x: 1 if x >= median_label else 0)
    else:
        if binarize_threshold is not None:
            dataset["label"] = dataset["label"].apply(lambda x: 1 if x >= binarize_threshold else 0)
    
    if resample:
        ros = SMOTE(sampling_strategy=0.67, random_state=42)
        X = dataset[dataset.columns[:-1]]
        y = dataset["label"]
        X_resampled, y_resampled = ros.fit_resample(X, y)
        dataset = pd.concat([X_resampled, y_resampled], axis=1)

    subjects = dataset.iloc[:, 0]
    features = dataset.iloc[:, 1:-1]
    labels = dataset.iloc[:, -1]

    return subjects, features, labels


if __name__ == "__main__":
    subjects, features, labels = load_dataset(normalized=False, apply_normalization=True)
