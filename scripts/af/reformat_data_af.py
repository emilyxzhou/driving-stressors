import datetime
import glob
import neurokit2 as nk
import numpy as np
import os
import pandas as pd
import sys
import warnings
sys.path.append("/home/emilyzho/distracted-driving/src")
warnings.filterwarnings("ignore", category=nk.NeuroKitWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from tqdm import tqdm


from constants_af import *
from data_reader_dd import get_data_for_subject, get_stimulus_data_for_subject
from feature_extractor import get_sampling_rate, get_eda_features, get_mean_signal

pd.set_option("future.no_silent_downcasting", True)

START = "Z_Start"
END = "Z_End.1"


METHOD_DICT = {
    SignalTypes.BR: get_mean_signal,
    SignalTypes.HR: get_mean_signal,
    SignalTypes.EDA: get_eda_features
}


def get_e4_session_indices(subject, session):
    e4_session_indices_file = "/media/data/public-data/drive/affectiveroad/AffectiveROAD_Data/Database/E4/Annot_E4_Left.csv"
    e4_session_indices_all = pd.read_csv(e4_session_indices_file)
    e4_session_indices = e4_session_indices_all[e4_session_indices_all["Drive-id"] == f"Drv{subject}"]
    # First half
    start = f"{session}_Start"
    end = f"{session}_End"
    start_index_1 = e4_session_indices[start].iloc[0]
    end_index_1 = e4_session_indices[end].iloc[0]
    # Second half
    start = f"{session}_Start.1"
    end = f"{session}_End.1"
    start_index_2 = e4_session_indices[start].iloc[0]
    end_index_2 = e4_session_indices[end].iloc[0]

    return [[start_index_1, end_index_1], [start_index_2, end_index_2]]


def get_bio_session_indices(subject, session):
    bio_session_indices_file = "/media/data/public-data/drive/affectiveroad/AffectiveROAD_Data/Database/Bioharness/Annot_Bioharness.csv"
    bio_session_indices_all = pd.read_csv(bio_session_indices_file)
    bio_session_indices = bio_session_indices_all[bio_session_indices_all["Drive_id"] == f"Drv{subject}"]
    # First half
    start = f"{session}_Start"
    end = f"{session}_End"
    start_index_1 = bio_session_indices[start].iloc[0]
    end_index_1 = bio_session_indices[end].iloc[0]
    # Second half
    start = f"{session}_Start.1"
    end = f"{session}_End.1"
    start_index_2 = bio_session_indices[start].iloc[0]
    end_index_2 = bio_session_indices[end].iloc[0]
    
    return [[start_index_1, end_index_1], [start_index_2, end_index_2]]


def get_annotation_indices(subject, session):
    annot_session_indices_file = os.path.join(ANNOTATIONS_FOLDER, "Annot_Subjective_metric.csv")
    annot_session_indices_all = pd.read_csv(annot_session_indices_file)
    annot_session_indices = annot_session_indices_all[annot_session_indices_all["Drive_id"] == f"Drv{subject}"]
    # First half
    start = f"{session}_Start"
    end = f"{session}_End"
    start_index_1 = annot_session_indices[start].iloc[0]
    end_index_1 = annot_session_indices[end].iloc[0]
    # Second half
    start = f"{session}_Start.1"
    end = f"{session}_End.1"
    start_index_2 = annot_session_indices[start].iloc[0]
    end_index_2 = annot_session_indices[end].iloc[0]
    
    return [[start_index_1, end_index_1], [start_index_2, end_index_2]]


def get_start_times(subject):
    # Annotations
    annot_session_indices_file = "/media/data/public-data/drive/affectiveroad/AffectiveROAD_Data/Database/Subj_metric/Annot_Subjective_metric.csv"
    annot_session_indices_all = pd.read_csv(annot_session_indices_file)
    annot_session_indices = annot_session_indices_all[annot_session_indices_all["Drive_id"] == f"Drv{subject}"]
    print("Annotation session indices " + "-"*23)
    print(annot_session_indices.to_string())

    # Bioharness annotations
    bio_session_indices_file = "/media/data/public-data/drive/affectiveroad/AffectiveROAD_Data/Database/Bioharness/Annot_Bioharness.csv"
    bio_session_indices_all = pd.read_csv(bio_session_indices_file)
    bio_session_indices = bio_session_indices_all[bio_session_indices_all["Drive_id"] == f"Drv{subject}"]
    print("\nBio session indices " + "-"*30)
    print(bio_session_indices.to_string())

    # E4 annotations
    e4_session_indices_file = "/media/data/public-data/drive/affectiveroad/AffectiveROAD_Data/Database/E4/Annot_E4_Left.csv"
    e4_session_indices_all = pd.read_csv(e4_session_indices_file)
    e4_session_indices = e4_session_indices_all[e4_session_indices_all["Drive-id"] == f"Drv{subject}"]
    print("\nE4 session indices " + "-"*31)
    print(e4_session_indices.to_string())

    # Bioharness
    file_path = os.path.join(BIOHARNESS_FOLDER, f"Bio_Drv{subject}.csv")
    bioharness = pd.read_csv(file_path, sep=";")
    bio_start_time = bioharness["Time"].iloc[0]
    bio_end_time = bioharness["Time"].iloc[-1]
    print("\nBio collection start time")
    print(bio_start_time)
    print("Bio collection end time")
    print(bio_end_time)

    # E4
    file_path = os.path.join(E4_FOLDER, f"{subject}-E4-Drv{subject}/Left/EDA.csv")
    eda = pd.read_csv(file_path, header=None)
    unix_timestamp = eda.iloc[0, 0]
    base_timestamp = datetime.datetime.fromtimestamp(int(unix_timestamp))
    print("E4 sync time")
    print(base_timestamp)

    file_path = os.path.join(E4_FOLDER, f"{subject}-E4-Drv{subject}/Left/tags.csv")
    e4_session_timestamps = pd.read_csv(file_path, header=None)
    e4_session_timestamps[0] = e4_session_timestamps[0].apply(lambda t: datetime.datetime.fromtimestamp(int(t)).time())
    print("\nE4 session timestamps " + "-"*28)
    print(e4_session_timestamps)

    e4_session_times = e4_session_indices.iloc[:, 1:]
    e4_session_times = e4_session_times.map(lambda t: (base_timestamp + datetime.timedelta(seconds=t//4)).time())
    print("\nE4 session start and end times " + "-"*28)
    print(e4_session_times.to_string())


def process_bioharness(subject, session, window=30, shift=1, sr=1):
    indices = get_bio_session_indices(subject, session)

    # Get Bioharness HR, BR data
    file_path = os.path.join(BIOHARNESS_FOLDER, f"Bio_Drv{subject}.csv")
    bioharness = pd.read_csv(file_path, sep=";")
    bioharness = bioharness[["Time", "HR", "BR"]]
    hr = bioharness[["Time", "HR"]]
    br = bioharness[["Time", "BR"]]

    features = {}
    for index_pair in indices:
        start_index = index_pair[0]
        end_index = index_pair[1]
        hr_segment = hr["HR"].iloc[start_index:end_index]
        br_segment = br["BR"].iloc[start_index:end_index]

        max_len = hr_segment.shape[0]
        offset = (window // 2) * sr
        center = offset
        while center + offset < max_len:
            segment = hr_segment.iloc[center-offset : center+offset].to_numpy().astype(float)
            out = get_mean_signal(segment, sr, "hr")
            # Add extracted features to list
            for feature_type in out.keys():
                if feature_type not in features.keys():
                    features[feature_type] = [out[feature_type]]
                else:
                    features[feature_type].append(out[feature_type])

            segment = br_segment.iloc[center-offset : center+offset].to_numpy().astype(float)
            out = get_mean_signal(segment, sr, "br")
            out["br"] = 1/out["br"] if out["br"] != 0 else np.nan
            # Add extracted features to list
            for feature_type in out.keys():
                if feature_type not in features.keys():
                    features[feature_type] = [out[feature_type]]
                else:
                    features[feature_type].append(out[feature_type])

            center += sr*shift

    features = pd.DataFrame(features)
    return features


def process_e4(subject, session, window=30, shift=1, sr=4):
    indices = get_e4_session_indices(subject, session)

    # Get E4 EDA data
    file_path = os.path.join(E4_FOLDER, f"{subject}-E4-Drv{subject}/Left/EDA.csv")
    eda = pd.read_csv(file_path, header=None)
    unix_timestamp = eda.iloc[0, 0]
    base_timestamp = datetime.datetime.fromtimestamp(int(unix_timestamp))
    eda = eda.iloc[2:, :]    # Drop timestamp and sampling rate rows

    timestep = 1 / sr
    index_timestamp = [
        base_timestamp + i * datetime.timedelta(seconds=timestep)
        for i in np.arange(eda.shape[0])
    ]
    eda.insert(0, "Time", index_timestamp)
    eda = eda.rename(columns={0: "EDA"})
    eda["Time"] = pd.to_datetime(eda["Time"])
    eda = eda.drop_duplicates()

    features = {}
    for index_pair in indices:
        start_index = index_pair[0]
        end_index = index_pair[1]
        eda_segment = eda["EDA"].iloc[start_index:end_index]

        max_len = eda.shape[0]
        offset = int(window / 2 * sr)
        center = offset
        while center + offset < max_len:
            segment = eda_segment.iloc[center-offset : center+offset].to_numpy().astype(float)
            if len(segment) < 20:    # Less than 5 seconds of data
                center += sr*shift
                continue
            out = get_eda_features(segment, sr)
            # Add extracted features to list
            for feature_type in out.keys():
                if feature_type not in features.keys():
                    features[feature_type] = [out[feature_type]]
                else:
                    features[feature_type].append(out[feature_type])

            center += sr*shift

    features = pd.DataFrame(features)
    return features


def process_annotations(subject, session, window=30, shift=1, sr=4):
    indices = get_annotation_indices(subject, session)

    # Get annotations
    file_path = os.path.join(ANNOTATIONS_FOLDER, f"SM_Drv{subject}.csv")
    annot = pd.read_csv(file_path, header=None).iloc[1:, :]
    
    features = {}
    for index_pair in indices:
        start_index = index_pair[0]
        end_index = index_pair[1]
        annot_segment = annot.iloc[start_index:end_index]

        max_len = annot.shape[0]
        offset = int(window / 2 * sr)
        center = offset
        while center + offset < max_len:
            segment = annot_segment.iloc[center-offset : center+offset].to_numpy().astype(float)
            if len(segment) < 20:    # Less than 5 seconds of data
                center += sr*shift
                continue
            out = get_mean_signal(segment, sr, "label")
            # Add extracted features to list
            for feature_type in out.keys():
                if feature_type not in features.keys():
                    features[feature_type] = [out[feature_type]]
                else:
                    features[feature_type].append(out[feature_type])

            center += sr*shift

    features = pd.DataFrame(features)
    return features


def make_dataset(normalize=True, scale=True):
    dataset = []
    labels = []
    for subject in SUBJECTS:
        for experiment_type in ExperimentTypes.DRIVE_EXPERIMENTS:
            file_name = os.path.join(
                DATA_PROCESSED_FOLDER, str(subject), 
                f"{subject}_{experiment_type}.csv"
            )
            try:
                features = pd.read_csv(file_name, index_col=0)
            except Exception:
                print(f"No data for {subject}, {experiment_type}")
                continue
            
            if normalize and not scale:
                baseline_file_name = os.path.join(
                    DATA_PROCESSED_FOLDER, str(subject), 
                    f"{subject}_Rest.csv"
                )
                baseline = pd.read_csv(baseline_file_name, index_col=0).iloc[0:30, :]
                mean, std = baseline.mean(axis=0), baseline.std(axis=0)
                features = features.sub(mean, axis=1)
            elif normalize and scale:
                baseline_file_name = os.path.join(
                    DATA_PROCESSED_FOLDER, str(subject), 
                    f"{subject}_Rest.csv"
                )
                baseline = pd.read_csv(baseline_file_name, index_col=0).iloc[0:30, :]
                mean, std = baseline.mean(axis=0), baseline.std(axis=0)
                features = features.sub(mean, axis=1)
                features = features.divide(std, axis=1)

            features.insert(0, "Subject", subject)
            dataset.append(features)

            feature_len = features.shape[0]

            # Get labels
            annotations_file = os.path.join(DATA_PROCESSED_FOLDER, str(subject), f"{subject}_{experiment_type}_annotations.csv")
            y = pd.read_csv(annotations_file, index_col=0)
            if y.shape[0] > feature_len:
                start = (y.shape[0] - feature_len) // 2
                end = start + feature_len
                y = y.iloc[start:end]
            labels.append(y)
    
    dataset = pd.concat(dataset, axis=0).reset_index(drop=True)
    labels = pd.concat(labels, axis=0).reset_index(drop=True)
    dataset = pd.concat([dataset, labels], axis=1)
    dataset = dataset.dropna(how="any", axis=0).reset_index(drop=True)
    
    if normalize and not scale:
        save_path = os.path.join(DATA_PROCESSED_FOLDER, "dataset_normalized.csv")
    elif normalize and scale:
        save_path = os.path.join(DATA_PROCESSED_FOLDER, "dataset_norm_scaled.csv")
    else:
        save_path = os.path.join(DATA_PROCESSED_FOLDER, "dataset.csv")
    dataset.to_csv(save_path)


if __name__ == "__main__":
    # for subject in SUBJECTS:
    #     print(f"Subject {subject} " + "-"*80)
        
    #     for experiment_type in (ExperimentTypes.EXPERIMENT_TYPES):
    #         print(f"{experiment_type} " + "-"*30)
    #         # Bioharness
    #         bio_features = process_bioharness(subject, experiment_type)

    #         # E4
    #         e4_features = process_e4(subject, experiment_type)

    #         features = pd.concat([bio_features, e4_features], axis=1)

    #         save_folder = os.path.join(
    #             DATA_PROCESSED_FOLDER, str(subject)
    #         )
    #         if not os.path.exists(save_folder):
    #             os.makedirs(save_folder)
            
    #         save_path = os.path.join(save_folder, f"{subject}_{experiment_type}.csv")
    #         features.to_csv(save_path)

    #         # Annotations
    #         if experiment_type in ExperimentTypes.DRIVE_EXPERIMENTS:
    #             annot = process_annotations(subject, experiment_type)
    #             save_path = os.path.join(save_folder, f"{subject}_{experiment_type}_annotations.csv")
    #             annot.to_csv(save_path)

    # make_dataset(normalize=False)
    # make_dataset(normalize=True, scale=False)
    make_dataset(normalize=True, scale=True)
