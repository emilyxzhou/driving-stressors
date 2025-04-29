SUBJECTS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
]

ANNOTATIONS_FOLDER = "/media/data/public-data/drive/affectiveroad/AffectiveROAD_Data/Database/Subj_metric"

BIOHARNESS_FOLDER = "/media/data/public-data/drive/affectiveroad/AffectiveROAD_Data/Database/Bioharness"
E4_FOLDER = "/media/data/public-data/drive/affectiveroad/AffectiveROAD_Data/Database/E4"
BIOHARNESS_ANNOTATIONS = "emilyzho@erie:/media/data/public-data/drive/affectiveroad/AffectiveROAD_Data/Database/Bioharness/Annot_Bioharness.csv"
E4_ANNOTATIONS_LEFT = "/media/data/public-data/drive/affectiveroad/AffectiveROAD_Data/Database/E4/Annot_E4_Left.csv"
E4_ANNOTATIONS_RIGHT = "/media/data/public-data/drive/affectiveroad/AffectiveROAD_Data/Database/E4/Annot_E4_Right.csv"

DATA_PROCESSED_FOLDER = "/media/data/public-data/drive/affectiveroad-processed/"

class SignalTypes:
    ACC = "ACC"
    BVP = "BVP"
    EDA = "EDA"
    HR = "HR"
    IBI = "IBI"
    BR = "BR"
    POST = "Posture"
    ACT = "Activity"

    BIOHARNESS_SIGNALS = [
        HR, BR, POST, ACT
    ]

    E4_SIGNALS = [
        ACC, BVP, EDA, HR, IBI
    ]

    SAMPLING_RATES = {
        EDA: 4,
        BR: 1,
        HR: 1
    }


class ExperimentTypes:
    REST = "Rest"
    # Z = "Z"
    CITY_1 = "City1"
    HWY = "Hwy"
    CITY_2 = "City2"

    EXPERIMENT_TYPES = [REST, CITY_1, HWY, CITY_2]

    DRIVE_EXPERIMENTS = [CITY_1, HWY, CITY_2]

    STRESS = []
    NON_STRESS = []