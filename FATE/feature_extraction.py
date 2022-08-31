import numpy as np
import pandas as pd
import os

def get_windows(data_df):
    data_gb = data_df.groupby()

def mean_and_polynomial_fitting(data_set):
    mean_trends = []
    for window in data

for i in range(1, 5):

    data_base = "/vol/bitbucket/hdr21/rul-prediction-fed/federated-learning/FL-data/decision-trees/fd00" + str(i) + "/scaled/"
    num_workers = ["2", "3", "5"]
    combinations = {
        "2": ["50-50", "70-30", "90-10"],
        "3": ["40-30-30", "50-40-10", "70-20-10"],
        "5": ["20-20-20-20-20", "40-30-10-10-10", "60-10-10-10-10"]
    }

    for n in num_workers:
        for c in combinations[n]:
