import glob
import numpy as np
import os
import pandas as pd
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, "src")
sys.path.append(src_dir)


from constants_dd import *
from data_reader_dd import get_data_for_subject, get_stimulus_data_for_subject
from feature_extractor import get_sampling_rate, get_eda_features, get_mean_signal

pd.set_option("future.no_silent_downcasting", True)


METHOD_DICT = {
    SignalTypes.BR: get_mean_signal,
    SignalTypes.HR: get_mean_signal,
    SignalTypes.EDA_PALM: get_eda_features
}


def reformat_and_save_data(subject, window=30, shift=1):
    # for experiment_type in ExperimentTypes.EXPERIMENT_TYPES:
    for experiment_type in [ExperimentTypes.FAILURE_LOADED, ExperimentTypes.FAILURE_NONLOADED]:
        save_folder = os.path.join(
            DATA_PROCESSED_FOLDER, str(subject)
        )

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_path = os.path.join(save_folder, f"P{subject}_{experiment_type}_physio.npy")
        save_path_csv = os.path.join(save_folder, f"P{subject}_{experiment_type}_physio.csv")

        feature_dfs = [] 
        for signal_type in SignalTypes.SIGNAL_TYPES:
            method = METHOD_DICT[signal_type]
            df = get_data_for_subject(subject, experiment_type, signal_type)

            if df is None:
                print(f"No data found for subject {subject}, {experiment_type}, {signal_type}")
                continue

            sr = int(get_sampling_rate(df))

            features = {}
            max_len = df.shape[0]
            offset = (window // 2) * sr
            center = offset
            while center + offset < max_len:
                for column in df.columns[1:]:
                    segment = df[column].iloc[center-offset : center+offset].to_numpy().astype(float)
                    out = method(segment, sr, signal_type)

                    # Add extracted features to list
                    for feature_type in out.keys():
                        if feature_type not in features.keys():
                            features[feature_type] = [out[feature_type]]
                        else:
                            features[feature_type].append(out[feature_type])

                    center += sr*shift

            features = pd.DataFrame(features)
            feature_dfs.append(features)


        if len(feature_dfs) > 0:
            feature_dfs = pd.concat(feature_dfs, axis=1)
            feature_dfs.to_csv(save_path_csv)

            feature_dfs = feature_dfs.to_numpy()
            np.save(save_path, feature_dfs)


def reformat_and_save_driving_signals(subject):
    for experiment_type in ExperimentTypes.EXPERIMENT_TYPES:
        save_folder = os.path.join(
            DATA_PROCESSED_FOLDER, str(subject)
        )

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_path = os.path.join(save_folder, f"P{subject}_{experiment_type}_driving.npy")
        save_path_csv = os.path.join(save_folder, f"P{subject}_{experiment_type}_driving.csv")

        df = get_data_for_subject(subject, experiment_type, SignalTypes.DRIVING)
        if df is None:
            print(f"No driving signal file found for subject {subject}, {experiment_type}")
            continue

        df.to_csv(save_path_csv)

        df = df.to_numpy()
        np.save(save_path, df)


def make_dataset(normalize=True):
    dataset = []
    labels = []
    for subject in SUBJECTS:
        for experiment_type in ExperimentTypes.EVALUATION:
            file_name = os.path.join(
                DATA_PROCESSED_FOLDER, str(subject), 
                f"P{subject}_{experiment_type}_physio.npy"
            )
            try:
                features_arr = np.load(file_name)
            except Exception:
                print(f"No data for P{subject}, {experiment_type}")
                continue
            features_arr = np.around(features_arr, decimals=3)
            
            # Some conditions are missing signals
            if features_arr.shape[1] < 7:
                print(f"Missing features for P{subject}, {experiment_type}")
                continue

            if normalize:
                try:
                    baseline_file_name = os.path.join(
                        DATA_PROCESSED_FOLDER, str(subject), 
                        f"P{subject}_{ExperimentTypes.BASELINE}_physio.npy"
                    )
                    baseline = np.load(baseline_file_name)
                    mean, std = baseline.mean(axis=0), baseline.std(axis=0)
                    features_arr = features_arr - mean
                except Exception:
                    print(f"Missing features for P{subject}, {ExperimentTypes.BASELINE}, skipping normalization")

            features_arr = np.insert(features_arr, 0, subject, axis=1)
            dataset.append(features_arr)

            # Generate labels
            if experiment_type in ExperimentTypes.STRESSORS:
                y = generate_labels_from_features(subject, experiment_type, features_arr)
            else:
                label = 0
                y = np.array([label for _ in range(features_arr.shape[0])]).reshape(-1, 1)
            labels.append(y)
    
    dataset = np.vstack(dataset)
    labels = np.vstack(labels)
    dataset = np.hstack([dataset, labels])
    
    if normalize:
        save_path = os.path.join(DATA_PROCESSED_FOLDER, "dataset_normalized.npy")
    else:
        save_path = os.path.join(DATA_PROCESSED_FOLDER, "dataset.npy")
    np.save(save_path, dataset)


def generate_labels_from_features(subject, experiment_type, features):
    labels = np.zeros((features.shape[0], 1))
    stimulus = get_stimulus_data_for_subject(subject, experiment_type)

    # Features are extracted from 30 second windows, starting at the 15th second
    for i in range(labels.shape[0]):
        # Stimuli start-stop ranges
        for j in range(stimulus.shape[0]):
            start = int(stimulus.iloc[j, 0])
            stop = int(stimulus.iloc[j, 1])
            if i + 15 in range(start, stop):
                labels[i] = 1
    return labels

if __name__ == "__main__":
    from tqdm import tqdm
    # for subject in tqdm(SUBJECTS):
    #     reformat_and_save_data(subject)
    make_dataset()