import csv
import datetime
import glob
import neurokit2 as nk
import numpy as np
import os
import pandas as pd

from tqdm import tqdm

from src.constants_trina import *
from src.data_reader_trina import get_features_for_subject
from src.feature_extractor import get_eda_features, get_hr_from_ecg, get_rsp_features_from_ecg, get_mean_signal


SAMPLE_RATE = 1/0.004


def get_signals_for_subject(subject):
    # This function extracts transient sensor data
    # and stores it in the signals dictionary
    file_path = os.path.join(DATA_BASE_FOLDER, f"{subject}_data.txt")
    df = pd.read_csv(file_path, sep="\t", header=19)  # Read txt data
    time = np.linspace(0,len(df.values[:,0])/SAMPLE_RATE, len(df.values[:,0])) # Seconds
    signals = {}                    # Data storage
    signals["time"] = time
    signals["ecg"] = df.values[:,1] # All ECG data
    signals["eda"] = df.values[:,2] # All EDA data
    signals["skt"] = df.values[:,3] # All EKT data
    signals["rsp"] = df.values[:,4] # All RSP data
    signals["ppg"] = df.values[:,5] # All PPG data
    signals = pd.DataFrame(signals)
    return time, signals


def extract_scene_data(subject):
    # This function extracts time stamps from laps
    # and stores in a dictionary
    file_path = os.path.join(DATA_BASE_FOLDER, f"{subject}_data.xlsx")
    df = pd.read_excel(file_path, sheet_name="Scenes")   # Read CSV Data
    laps = {}                                            # Data storage
    for lap in df.values:                                # Loop through laps
        info = {}                                        # Data storage for given lap
        info["start"] = datetime.datetime(int(lap[1]), int(lap[2]), int(lap[3]), int(lap[4]), int(lap[5]), int(lap[6])) # Start time
        info["end"] = datetime.datetime(int(lap[1]), int(lap[2]), int(lap[3]), int(lap[7]), int(lap[8]), int(lap[9]))   # End time
        info["events"] = []                              # Creates empty storage for later
        laps[str(lap[0])] = info                         # Append data
    return laps 


def extract_event_data(subject, laps):
    file_path = os.path.join(DATA_BASE_FOLDER, f"{subject}_data.xlsx")
    df = pd.read_excel(file_path, sheet_name="Events")    # Read CSV Data
    scenes = np.unique(df.values[:,0])
    for scene in scenes:                      # Loop through event flags
        if scene == "Start Time":
            event = df.values[0,:]
            start_time = datetime.datetime(event[2], event[3], event[4], event[5], event[6], event[7]) # Start time
        else:
            events = []                    # Data storage
            for event in df.values:        # Loop events
                if event[0] == scene:      # If event is for scene, save
                    events.append([event[1], datetime.datetime(event[2], event[3], event[4], event[5], event[6], event[7])]) # Event time
            laps[scene]["events"] = events # Append all event to scene

    return laps, start_time


def pack_data(subject):
    # This pulls data from excel and stores by participant. 
    laps = extract_scene_data(subject)                    # Extract lap timeframe
    laps, start_time = extract_event_data(subject, laps)  # Extract event flags
    time, signals = get_signals_for_subject(subject)         # Extract raw data
    participant = {}                                      # Create storage
    participant["name"] = subject                         # Participant name
    participant["start_time"] = start_time                # When data collection starts
    participant["time"] = time                            # Save time 
    participant["signals"] = signals                      # Save signals
    participant["laps"] = laps                            # Lap time stamps
    return participant


def convert_time_to_index(current_time, start_time):
    # This simple script takes a date and time
    # and finds index in transient data
    time_change = current_time - start_time         # difference between event and start time
    time_change = time_change.total_seconds()       # convert to seconds
    time_change = np.floor(time_change*SAMPLE_RATE) # convert to index of data
    return int(time_change)                         # Return time index


def load_subject(subject):
    return pack_data(subject)


def extract_features(signals, window=30, shift=1):
    """
    signals: DataFrame containing time index and signal columns

    Extracts hr, rsp_rate, scl_mean, scl_slope, scr_peaks, scr_amp, scr_rise 
    """
    features = {}

    max_len = signals.shape[0]
    offset = int(window / 2 * SAMPLE_RATE)
    center = offset

    while center + offset < max_len:
        segment = signals.iloc[center-offset : center+offset, :]
        if len(segment) < 7500:    # Less than 30 seconds of data
            center += SAMPLE_RATE*shift
            continue

        # hr
        out = get_hr_from_ecg(segment["ecg"], SAMPLE_RATE)
        # Add extracted features to list
        for feature_type in out.keys():
            if feature_type not in features.keys():
                features[feature_type] = [out[feature_type]]
            else:
                features[feature_type].append(out[feature_type])

        # rsp_rate
        out = get_rsp_features_from_ecg(segment["ecg"], segment["rsp"], SAMPLE_RATE)
        # Add extracted features to list
        for feature_type in out.keys():
            if feature_type not in features.keys():
                features[feature_type] = [out[feature_type]]
            else:
                features[feature_type].append(out[feature_type])

        # eda
        out = get_eda_features(segment["eda"], SAMPLE_RATE)
        # Add extracted features to list
        for feature_type in out.keys():
            if feature_type not in features.keys():
                features[feature_type] = [out[feature_type]]
            else:
                features[feature_type].append(out[feature_type])
        
        center += int(SAMPLE_RATE*shift)

    features = pd.DataFrame(features)
    return features


def generate_and_save_features(subject):   
    """
    Free roam: baseline
    Averages features extracted from 30-second segments, with a 1-second sliding window.
    Starts at 15 seconds
    """
    print(f"{subject} " + "-"*60)
    data = load_subject(subject)
    signals = data["signals"]
    segments = data["laps"]
    drive_start_time = data["start_time"]

    # Baseline: ------------------------------
    print("Free drive")
    free_drive = segments["Free Roam"]
    start = free_drive["start"]
    end = free_drive["end"]
    start_index = convert_time_to_index(start, drive_start_time)
    end_index = convert_time_to_index(end, drive_start_time)
    free_drive_signals = signals.iloc[start_index:end_index, :].reset_index(drop=True)
    
    free_drive_features = extract_features(free_drive_signals)
    save_path = os.path.join(DATA_PROCESSED_FOLDER, f"{subject}_baseline.csv")
    free_drive_features.to_csv(save_path)

    for name in segments.keys():
        if name == "Free Roam":
            continue
        print(name)
        segment = segments[name]

        start = segment["start"]
        end = segment["end"]
        start_index = convert_time_to_index(start, drive_start_time)
        end_index = convert_time_to_index(end, drive_start_time)

        signal_segment = signals.iloc[start_index:end_index, :].reset_index(drop=True)
        features = extract_features(signal_segment)
        save_path = os.path.join(DATA_PROCESSED_FOLDER, f"{subject}_{name}.csv")
        features.to_csv(save_path)


def generate_and_save_labels(subject):
    # Note: Features start at 15s 
    data = load_subject(subject)
    segments = data["laps"]

    for name in segments.keys():
        if name == "Free Roam":
            features = get_features_for_subject(subject, "baseline")
            labels = {"label": [0 for _ in range(features.shape[0])]}
            labels = pd.DataFrame(labels)
        else:
            features = get_features_for_subject(subject, name)

            segment = segments[name]

            segment_start = segment["start"]
            events = segment["events"]
            event_indices = []

            for event in events:
                if event[0] in STRESSORS:
                    start = event[1]
                    end = start + datetime.timedelta(seconds=15)
                    start_index = convert_time_to_index(start, segment_start) // 250 - 15    # 15s offset from feature extraction
                    end_index = convert_time_to_index(end, segment_start) // 250 - 15
                    if start_index < 0:
                        start_index = 0
                    if end_index < 0:
                        end_index = 0
                    event_indices.append((start_index, end_index))
            
            labels = np.array([0 for _ in range(features.shape[0])])
            for indices in event_indices:
                start = indices[0]
                end = indices[1]
                labels[start:end] = 1
            
            labels = {"label": labels}
            labels = pd.DataFrame(labels)

        if name == "Free Roam":
            save_path = os.path.join(DATA_PROCESSED_FOLDER, f"{subject}_baseline_labels.csv")
        else:
            save_path = os.path.join(DATA_PROCESSED_FOLDER, f"{subject}_{name}_labels.csv")

        labels.to_csv(save_path)


def make_dataset(normalize="z"):
    """
    normalize: None, "z", or "median"
    """
    dataset = []
    labels = []
    for subject in SUBJECTS:
        for experiment_type in ExperimentTypes.DRIVE_EXPERIMENTS:
            file_name = os.path.join(
                DATA_PROCESSED_FOLDER, 
                f"{subject}_{experiment_type}.csv"
            )
            try:
                features = pd.read_csv(file_name, index_col=0)
            except Exception:
                print(f"No data for {subject}, {experiment_type}")
                continue
            
            if normalize == "z":
                baseline_file_name = os.path.join(
                    DATA_PROCESSED_FOLDER,
                    f"{subject}_baseline.csv"
                )
                baseline = pd.read_csv(baseline_file_name, index_col=0).iloc[0:60, :]
                mean, std = baseline.mean(axis=0), baseline.std(axis=0)
                features = features.sub(mean, axis=1)
                features = features.divide(std)

            elif normalize == "median":
                baseline_file_name = os.path.join(
                    DATA_PROCESSED_FOLDER,
                    f"{subject}_baseline.csv"
                )
                baseline = pd.read_csv(baseline_file_name, index_col=0).iloc[0:60, :]
                med = baseline.median(axis=0)
                q1 = baseline.quantile(0.25, axis=0)
                q3 = baseline.quantile(0.75, axis=0)
                iqr = q3.sub(q1)
                features = features.sub(med, axis=1)
                features = features.divide(iqr)

            features.insert(0, "Subject", subject)
            dataset.append(features)

            feature_len = features.shape[0]

            # Get labels
            annotations_file = os.path.join(DATA_PROCESSED_FOLDER, f"{subject}_{experiment_type}_labels.csv")
            y = pd.read_csv(annotations_file, index_col=0)
            labels.append(y)
    
    dataset = pd.concat(dataset, axis=0).reset_index(drop=True)
    labels = pd.concat(labels, axis=0).reset_index(drop=True)
    dataset = pd.concat([dataset, labels], axis=1)
    # dataset = dataset.dropna(how="any", axis=0).reset_index(drop=True)
    
    if normalize == "z":
        save_path = os.path.join(DATA_PROCESSED_FOLDER, "dataset_z.csv")
    elif normalize == "median":
        save_path = os.path.join(DATA_PROCESSED_FOLDER, "dataset_med.csv")
    else:
        save_path = os.path.join(DATA_PROCESSED_FOLDER, "dataset.csv")
    dataset.to_csv(save_path)


if __name__ == "__main__":
    for subject in tqdm(SUBJECTS):
    #     generate_and_save_features(subject)
        generate_and_save_labels(subject)
    
    make_dataset(normalize=False)
    make_dataset(normalize="z")
    make_dataset(normalize="median")

    # View event names
    # for subject in SUBJECTS[0:1]:
    #     data = load_subject(subject)
    #     laps = data["laps"]
    #     for key in laps.keys():
    #         print(f"{key} " + "-"*50)
    #         events = laps[key]["events"]
    #         print([event[0] for event in events])