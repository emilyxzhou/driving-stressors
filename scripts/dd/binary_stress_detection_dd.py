import numpy as np
import os
import pandas as pd
import random
import sys
sys.path.append("/home/emilyzho/distracted-driving/src/")

from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from tqdm import tqdm
from xgboost import XGBClassifier

from constants_dd import *
from data_reader_dd import get_data_for_subject, load_dataset, segment_data

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def loso_binary_classification(model, standardize="subject", random_seed=42):
    """
    Parameters ---------------
    :param standardize: "subject" or "dataset"
    :type standardize: str
    """
    random.seed(random_seed)
    # Load dataset 
    dataset = load_dataset()
    dataset = dataset.drop("eda_palm", axis=1)
    x = dataset.loc[:, dataset.columns != "label"]
    y = dataset.loc[:, ["label"]]

    # Randomly select 20% of subjects for the held-out test set
    # num_train = int(len(SUBJECTS) * 0.8)
    # subjects = random.sample(SUBJECTS, num_train)
    subjects = SUBJECTS
    train_indices = x[x["ID"].isin(subjects)].index.tolist()

    train_x = x.iloc[train_indices, :].reset_index(drop=True)
    train_y = y.iloc[train_indices, :].reset_index(drop=True)

    test_x = x[~x.index.isin(train_indices)].reset_index(drop=True)
    test_x = test_x.drop("ID", axis=1)
    test_y = y[~y.index.isin(train_indices)].reset_index(drop=True)

    cv_results = {
        "acc": [],
        "f1": [],
        "auc": []
    }

    # test_results = {
    #     "acc": [],
    #     "f1": [],
    #     "auc": []
    # }

    scaler = StandardScaler().set_output(transform="pandas")

    # LOSO-CV
    for subject in tqdm(subjects):
        rus = RandomUnderSampler(random_state=random_seed, sampling_strategy=2/3)
        train_indices = train_x[train_x["ID"] != subject].index.tolist()
        
        cv_train_x = train_x.iloc[train_indices, :].reset_index(drop=True)
        cv_train_y = train_y.iloc[train_indices, :].reset_index(drop=True)
        val_x = train_x[~train_x.index.isin(train_indices)].reset_index(drop=True)
        val_y = train_y[~train_y.index.isin(train_indices)].reset_index(drop=True)

        # Undersample majority class
        cv_train_x, cv_train_y = rus.fit_resample(cv_train_x, cv_train_y)
        cv_train_x = cv_train_x.reset_index(drop=True)
        cv_train_y = cv_train_y.reset_index(drop=True)

        if val_x.shape[0] == 0:
            print(f"No data for subject {subject}, skipping")
            continue
        
        # Subject-wise z-score normalization
        if standardize == "subject":
            cv_train_x_scaled = []
            val_x_scaled = []
            for s in subjects:  
                if s != subject:
                    subject_rows = cv_train_x.index[cv_train_x["ID"] == s]
                    transformed = scaler.fit_transform(cv_train_x.iloc[subject_rows, :].drop("ID", axis=1)).reset_index(drop=True)
                    cv_train_x_scaled.append(transformed)
                else:
                    subject_rows = val_x.index[val_x["ID"] == s]
                    transformed = scaler.fit_transform(val_x.iloc[subject_rows, :].drop("ID", axis=1)).reset_index(drop=True)
                    val_x_scaled.append(transformed)

            cv_train_x_scaled = pd.concat(cv_train_x_scaled, axis=0).reset_index(drop=True)
            val_x_scaled = pd.concat(val_x_scaled, axis=0).reset_index(drop=True)
        # Normalize over entire training set
        elif standardize == "dataset":
            cv_train_x_scaled = scaler.fit_transform(cv_train_x)
            val_x_scaled = scaler.transform(val_x)
        # No standardization
        else:
            cv_train_x_scaled = cv_train_x
            val_x_scaled = val_x

        # print(cv_train_y.value_counts())

        if val_x.shape[0] > 0:
            model.fit(cv_train_x_scaled, cv_train_y.values.ravel())
            y_pred = model.predict(val_x_scaled)
            neg = np.count_nonzero(y_pred == 0)
            pos = np.count_nonzero(y_pred == 1)
            # print(f"0: {neg}, 1: {pos}")

            acc = accuracy_score(val_y, y_pred)
            f1 = f1_score(val_y, y_pred)
            auc = roc_auc_score(val_y, y_pred)

            cv_results["acc"].append(acc)
            cv_results["f1"].append(f1)
            cv_results["auc"].append(auc)

    # Evaluate on held-out test set
    # train_x = train_x.drop("ID", axis=1)
    # model.fit(train_x, train_y.values.ravel())
    # y_pred = model.predict(test_x)
    # acc = accuracy_score(test_y, y_pred)
    # f1 = f1_score(test_y, y_pred)
    # auc = roc_auc_score(test_y, y_pred)

    # test_results["acc"].append(acc)
    # test_results["f1"].append(f1)
    # test_results["auc"].append(auc)

    for key in cv_results.keys():
        cv_results[key] = np.nanmean(cv_results[key])

    # for key in test_results.keys():
    #     test_results[key] = np.nanmean(test_results[key])

    return model, cv_results, # test_results


def group_kfold_classification(model):
    dataset = load_dataset()
    group = dataset["ID"].tolist()
    gkf = GroupKFold(n_splits=10)



if __name__ == "__main__":
    results_path = os.path.join(
        DATA_PROCESSED_FOLDER, "results.csv"
    )
    models = {
        "SVM": SVC,
        "KNN": KNN,
        # "RF": RF,
        # "XGB": XGBClassifier()
    }
    params = {
        "SVM": [
            {"C": 1, "gamma": 10},
            {"C": 100, "gamma": 10},
            {"C": 100, "gamma": 20},
        ],
        "KNN": [
            {"n_neighbors": 5},
            {"n_neighbors": 10},
            {"n_neighbors": 45},
            {"n_neighbors": 60},
        ],
        # "RF": [
        #     {"n_estimators": 100},
        #     {"n_estimators": 150},
        #     {"n_estimators": 200}
        # ]
        # "XGB": [
        #     {},
        #     {}
        # ],
    }

    for model_name in models.keys():
        print(f"Model: {model_name} " + "-"*80)
            
        for i, param in enumerate(params[model_name]):
            print(f"Parameters: {param}")
            model = models[model_name](**param)
            model, cv_results = loso_binary_classification(model, standardize="dataset")

            acc = cv_results["acc"]
            f1 = cv_results["f1"]
            auc = cv_results["auc"]

            print("CV " + "-"*47)
            print(f"Accuracy: {acc}")
            print(f"F1: {f1}")
            print(f"AUC: {auc}")

            cv_results.update(param)
            cv_results = {key: [float(cv_results[key])] for key in cv_results.keys()}
            cv_results = pd.DataFrame(cv_results)
            save_path = os.path.join(DATA_PROCESSED_FOLDER, "results", f"{model_name}_{i}.csv")
            cv_results.to_csv(save_path)

            # print("Test " + "-"*45)
            # print(f"Accuracy: {test_results["acc"]}")
            # print(f"F1: {test_results["f1"]}")
            # print(f"AUC: {test_results["auc"]}")

            print("")