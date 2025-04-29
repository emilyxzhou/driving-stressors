import csv
import glob
import numpy as np
import os
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer
from tqdm import tqdm

from src.constants_dd import *


def get_data_for_subject(subject, experiment_type, signal_type):
    """
    Parameters ---------------
    :param subject: Subject ID
    :type subject: int
    :param experiment_type: Type of experimental condition (Baseline, Relaxing, Loaded, etc.)
    :type experiment_type: str
    :param signal_type: Type of signal (breathing, heart_rate, eda_palm, etc.)
    :type signal_type: str
    """
    experiment_folder = ExperimentTypes.EXPERIMENT_FOLDERS[experiment_type]
    signal_ext = SignalTypes.SIGNAL_EXT[signal_type]
    subject_files = glob.glob(f"{DATA_BASE_FOLDER}/T*0{subject}/*{experiment_folder}/*{signal_ext}")
    
    if len(subject_files) == 0:
        return None
    
    df = pd.read_excel(subject_files[0])
    df = df.iloc[8:].reset_index(drop=True)
    # Drop column containing frame #
    df = df.drop(df.columns[0], axis=1)
    
    if signal_type == SignalTypes.DRIVING:
        columns = ["timestamp"] + [Features.SPEED, Features.ACC, Features.BRAKE, Features.STEERING, Features.LANE]
        df = df.drop(df.columns[2], axis=1)
    else:
        columns = ["timestamp", signal_type]
    df.columns = columns

    return df


def get_stimulus_data_for_subject(subject, experiment_type):
    experiment_folder = ExperimentTypes.EXPERIMENT_FOLDERS[experiment_type]
    subject_files = glob.glob(f"{DATA_BASE_FOLDER}/T*0{subject}/*{experiment_folder}/*stm")
    
    if len(subject_files) == 0:
        return None

    df = pd.read_excel(subject_files[0])
    df = df.iloc[8:, [0, 1, 6]].reset_index(drop=True)
    df.columns = [
        "StartTime", "EndTime", "ActionType"
    ]
    return df


def load_dataset(normalize=True, interpolate=True, resample=True):
    if normalize:
        dataset = np.load(NORMALIZED_DATASET_PATH)
    else:
        dataset = np.load(DATASET_PATH)
        
    if interpolate:
        imputer = KNNImputer(n_neighbors=3)
        dataset[:, 1:-1] = imputer.fit_transform(dataset[:, 1:-1])
    else:
        dataset = dataset[~np.isnan(dataset).any(axis=1)]

    if resample:
        ros = SMOTE(sampling_strategy=0.67, random_state=42)
        X = dataset[:, :-1]
        y = dataset[:, [-1]]
        X_resampled, y_resampled = ros.fit_resample(X, y)
        y_resampled = y_resampled[:, np.newaxis]
        dataset = np.hstack([X_resampled, y_resampled])
    
    subjects = dataset[:, 0]
    features = dataset[:, 1:-1]    # first column = subjects, last column = labels
    labels = dataset[:, -1]
    return subjects, features, labels


if __name__ == "__main__":
    subjects, features, labels = load_dataset(interpolate=True)
    print(labels[1000:2000])