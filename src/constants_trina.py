import os

# DATA_BASE_FOLDER = "/media/data/toyota/processed_data_old/trina-bind/"
DATA_BASE_FOLDER = "/media/data/toyota/raw_data/trina_10"
DATA_PROCESSED_FOLDER = "/media/data/toyota/processed_data/trina_10_processed"
# DATASET_PATH = save_path = os.path.join(
#     DATA_PROCESSED_FOLDER,
#     "dataset.npy"
# )
# NORMALIZED_DATASET_PATH = save_path = os.path.join(
#     DATA_PROCESSED_FOLDER,
#     "dataset_normalized.npy"
# )

SUBJECTS = [
    "DI101228", "HS141262", "NY101177", "OL141172", 
    "PE141166", "QD092098", "QL151179", "SZ101195", 
    "WR111245", "WZ102091"
]

class ExperimentTypes:
    BASELINE = "baseline"
    CITY = "City"
    FOG = "Fog"
    CONSTRUCTION = "Construction"
    TRAFFIC = "Traffic"
    SURPRISE = "Surprise"

    EXPERIMENT_TYPES = [
        BASELINE, CITY, FOG, CONSTRUCTION, TRAFFIC, SURPRISE
    ]

    DRIVE_EXPERIMENTS = [
        CITY, FOG, CONSTRUCTION, TRAFFIC, SURPRISE
    ]


STRESSORS = [
    # City
    "Truck Appears", "Stop 1", "Stop 2", "Truck Clear", "Red Light 1", "Red Light 2", "Traffic", "City Speed Over",
    # Fog
    "Traffic 1", "Brake 1", "Brake 2",
    # Construction
    "Construction 1", "Traffic", "Construction 2"
    # Traffic
    "Traffic 1", "Traffic Clear", "Bump", "Traffic 2",
    # Surprise
    "Stop Sign", "Clear 1", "Event 2", "Stuck", "Bridge"
]