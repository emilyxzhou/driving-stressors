import csv
import datetime
import glob
import neurokit2 as nk
import numpy as np
import os
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.constants_trina import *


def get_features_for_subject(subject, segment_name):
    file = glob.glob(
        os.path.join(DATA_PROCESSED_FOLDER, f"{subject}_{segment_name}*")
    )[0]
    features = pd.read_csv(file, index_col=0)
    return features


def load_dataset(normalized=False, apply_normalization=False, resample=True, interpolate=True):
    if normalized == "z":
        file_path = os.path.join(DATA_PROCESSED_FOLDER, "dataset_z.csv")
        dataset = pd.read_csv(file_path, index_col=0)
    elif normalized == "median":
        file_path = os.path.join(DATA_PROCESSED_FOLDER, "dataset_med.csv")
        dataset = pd.read_csv(file_path, index_col=0)
    else:
        file_path = os.path.join(DATA_PROCESSED_FOLDER, "dataset.csv")
        dataset = pd.read_csv(file_path, index_col=0)

    dataset[dataset.columns[1:-1]] = dataset[dataset.columns[1:-1]].astype(float)

    if not normalized and apply_normalization:
        # Perform subject-wise normalization
        scaler = StandardScaler()
        subjects = dataset.iloc[:, 0].unique().tolist()
        for subject in subjects:
            rows = dataset.loc[dataset["Subject"] == subject].index.tolist()
            interpolated = scaler.fit_transform(dataset.iloc[rows, 1:-1])
            dataset.iloc[rows, 1:-1] = interpolated

    if interpolate:
        imputer = KNNImputer(n_neighbors=3)
        dataset.iloc[:, 1:-1] = pd.DataFrame(imputer.fit_transform(dataset.iloc[:, 1:-1]), columns=dataset.columns[1:-1]).astype(float)
    else:
        dataset = dataset.dropna(how="any", axis=0)
    
    if resample:
        ros = SMOTE(sampling_strategy=0.67, random_state=42)
        X = dataset[dataset.columns[:-1]]
        y = dataset["label"]
        X_resampled, y_resampled = ros.fit_resample(X, y)
        dataset = pd.concat([X_resampled, y_resampled], axis=1)

    subjects = dataset.iloc[:, 0]
    features = dataset.iloc[:, 1:-1]    # first column = subjects, last column = labels
    labels = dataset.iloc[:, -1]
    return subjects, features, labels


if __name__ == "__main__":
    subjects, features, labels = load_dataset(normalized=False, apply_normalization=True, resample=False, interpolate=True)
    # print(features.shape)
    # print(features.head(20))