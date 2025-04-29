import neurokit2 as nk
import numpy as np
import pandas as pd


def get_sampling_rate(signal):
    timestamps = signal["timestamp"]
    diff = timestamps.iloc[1] - timestamps.iloc[0]
    sr = 1/diff
    return sr


def get_eda_features(eda_rec, sr, signal_type=None):
    """
    Get EDA-derived features
    """
    eda, _ = nk.eda_process(eda_rec, sampling_rate=sr)

    scl = eda["EDA_Tonic"]
    scl_mean = np.mean(scl)
    scl_slope = np.polyfit(np.arange(len(scl)), scl, 1)[0]

    scr_count = eda["SCR_Peaks"].sum()

    scr_indices = np.where(eda["SCR_Peaks"])[0]
    scr_ratio = len(scr_indices) / len(eda_rec)
    scr_amp = eda["SCR_Amplitude"]
    scr_amp = scr_amp[scr_indices].mean()
    scr_rise = eda["SCR_RiseTime"]
    scr_rise = scr_rise[scr_indices].mean()

    out = {
        "scl_mean": scl_mean,
        "scl_slope": scl_slope,
        "scr_count": scr_count,
        "scr_ratio": scr_ratio,
        "scr_amp": scr_amp,
        "scr_rise": scr_rise,
    }

    return out


def get_hr_from_ecg(signal, sr, signal_type=None):
    ecg_processed, _ = nk.ecg_process(signal, sr)
    hr = ecg_processed["ECG_Rate"].mean()
    out = {"hr": hr}
    return out


def get_rsp_rate(signal, sr, signal_type=None):
    rsp_whole = nk.rsp_clean(signal, sampling_rate=sr)
    rsp_rate = nk.rsp_rate(rsp_whole, sampling_rate=sr).mean()
    out = {"rsp_rate": rsp_rate}
    return out


def get_rsp_features_from_ecg(ecg_signal, rsp_signal=None, sr=250, signal_type=None):
    ecg_processed, info = nk.ecg_process(ecg_signal, sampling_rate=sr)
    ecg_clean = ecg_processed["ECG_Clean"]

    if rsp_signal is not None:
        rsp_processed, rsp_info = nk.rsp_process(rsp_signal, sampling_rate=sr)
    else:
        # Extract peaks
        rpeaks, info = nk.ecg_peaks(ecg_clean, sampling_rate=sr)
        # Compute rate
        ecg_rate = nk.signal_rate(rpeaks, sampling_rate=sr, desired_length=len(rpeaks))
        edr = nk.ecg_rsp(ecg_rate, sampling_rate=sr)
        rsp_whole = nk.rsp_clean(edr, sampling_rate=sr)
        rsp_processed, rsp_info = nk.rsp_process(rsp_whole, sampling_rate=sr)

    troughs = rsp_info["RSP_Troughs"]
    if len(troughs) < 2:
        rsp_period = 10.0
    else:
        rsp_diff = np.diff(rsp_info["RSP_Troughs"])
        rsp_period = np.mean(rsp_diff) / sr

    rsp_depth = rsp_processed["RSP_Amplitude"].mean()
    rvt = rsp_processed["RSP_RVT"].mean()
    # rsa = nk.hrv_rsa(
    #     ecg_processed,
    #     rsp_processed,
    #     rpeaks,
    #     window=len(ecg_signal) // sr,
    #     sampling_rate=sr,
    #     continuous=False,
    # )
    # rsa = rsa["RSA_P2T_Mean"]

    out = {
        "rsp_period": rsp_period,
        "rsp_depth": rsp_depth,
        "rvt": rvt
    }
    return out


def get_mean_signal(signal, sr, signal_type):
    mean_val = np.mean(signal)
    out = {signal_type: mean_val}
    return out