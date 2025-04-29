import os

DATA_BASE_FOLDER = "/media/data/public-data/drive/distracted-driving/"
DATA_PROCESSED_FOLDER = "/media/data/public-data/drive/distracted-driving-processed/"
DATASET_PATH = save_path = os.path.join(
    DATA_PROCESSED_FOLDER,
    "dataset.npy"
)
NORMALIZED_DATASET_PATH = save_path = os.path.join(
    DATA_PROCESSED_FOLDER,
    "dataset_normalized.npy"
)
SUBJECTS = list(range(2, 9)) + [10] + [13, 14] + list(range(16, 28)) + [31] + list(range(33, 37)) + [38, 39] + list(range(42, 48)) + [50, 51, 54, 55, 60, 61, 62, 66, 68, 73] + list(range(75, 78)) + \
    list(range(79, 82)) + list(range(83, 85)) + [86]    
# Did not include subject 88 due to lack of data
# Subjects 11, 28, 40, 64, 82 were missing EDA signal
# Subjects 1, 9, 12, 15, 41, 64, 74, 88 were missing HR signal/bad HR signal


class ExperimentTypes:
    # BASELINE = "Baseline"    # Does not contain breathing and heart rate signals
    PRACTICE = "Practice"
    RELAXING = "Relaxing"
    LOADED = "Loaded"
    COGNITIVE = "Cognitive"
    EMOTIONAL = "Emotional"
    SENSORIMOTOR = "Sensorimotor"
    FAILURE_LOADED = "Failure_L"
    FAILURE_NONLOADED = "Failure_N"

    EXPERIMENT_TYPES = [
        # BASELINE, 
        PRACTICE, RELAXING,
        LOADED, COGNITIVE, EMOTIONAL, SENSORIMOTOR,
        FAILURE_LOADED, FAILURE_NONLOADED
    ]

    EXPERIMENT_FOLDERS = {
        # BASELINE: "BL",
        PRACTICE: "PD",
        RELAXING: "RD",
        LOADED: "ND",
        COGNITIVE: "CD",
        EMOTIONAL: "ED",
        SENSORIMOTOR: "MD",
        FAILURE_LOADED: "FDL",
        FAILURE_NONLOADED: "FDN"
    }

    BASELINE = PRACTICE
    EVALUATION = [RELAXING, LOADED, COGNITIVE, EMOTIONAL, SENSORIMOTOR]

    STRESSORS = [COGNITIVE, EMOTIONAL, SENSORIMOTOR, FAILURE_LOADED, FAILURE_NONLOADED]


class SignalTypes:
    BIO = "biographic"
    PSYCH = "psychometric"
    TRAIT = "trait psychometric"
    BR = "breathing_rate"
    HR = "heart_rate"
    EDA_PALM = "eda_palm"
    DRIVING = "driving"

    SIGNAL_TYPES = [BR, HR, EDA_PALM]
    SELF_REPORTS = [BIO, PSYCH, TRAIT]

    SIGNAL_EXT = {
        BIO: "b",
        PSYCH: "bar",
        TRAIT: "tp",
        BR: "BR",
        HR: "HR",
        EDA_PALM: "peda",
        DRIVING: "res"
    }


class Features:
    BR = "breathing_rate"
    HR = "heart_rate"
    SCL_MEAN = "scl_mean"
    SCL_SLOPE = "scl_slope"
    SCR_COUNT = "scr_count"
    SCR_RATIO = "scr_ratio"
    SCR_AMP = "scr_amp"
    SCR_RISE = "scr_rise"
    

    SPEED = "speed"
    ACC = "acceleration"
    BRAKE = "brake_force"
    STEERING = "steering"
    LANE = "lane_position"

    FEATURES = [BR, HR, SCL_MEAN, SCL_SLOPE, SCR_RATIO, SCR_AMP, SCR_RISE]