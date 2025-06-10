import random
import os
import time
import json
import warnings; warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from TSB_UAD.utils.slidingWindows import find_length
from TSB_UAD.vus.metrics import get_metrics

from TabPFNGLaSSDetector import TabPFNGLaSSDetector
from experiment_generator import experiments_generator
from db_utils import experiment_exists, insert_experiment_result
from experiment_configs import IFOREST_NAME, LOF_NAME, ONLINE_TAG


selected_domains = ['NAB', 'YAHOO']
nof_experiments_per_category = 3
public_root = "./data/"
normality_level_min = 2
normality_level_max = 4
random_seed = 42

experiments = experiments_generator(
    selected_domains=selected_domains,
    nof_experiments_per_category=nof_experiments_per_category,
    public_root=public_root,
    normality_level_min=normality_level_min,
    normality_level_max=normality_level_max,
    random_seed=random_seed
)

random.seed(None)
random.shuffle(experiments) # shuffling for executing from multiple devices in parallel

for experiment_idx, experiment_files in enumerate(experiments):
    print(f"Working on experiment {experiment_idx + 1} of {len(experiments)}: {experiment_files}")


    data, labels = [], []

    for file_path in experiment_files:
        df = pd.read_csv(file_path, header=None).dropna()
        series = df.iloc[:, 0].astype(float).to_numpy()
        label = df.iloc[:, 1].astype(int).to_numpy()
        data.extend(series)
        labels.extend(label)

    data = np.array(data)
    labels = np.array(labels)

    normality_levels = len(experiment_files)
    series_length = int(len(data))
    nof_anomalies = sum(labels)

    slidingWindow = find_length(data)

    for n_clusters in [10, 20]:
        for state_size in [1000, 4000]:
            extra_info = {
                'n_clusters': n_clusters,
                'state_size': state_size,
            }

            ### Online 1 - IForest with TabPFN
            modelName = IFOREST_NAME + "_TabPFN"
            tag = ONLINE_TAG
            if not experiment_exists(experiment_files, modelName, tag, extra_info):
                clf = TabPFNGLaSSDetector(
                    batch_frac=0.1,
                    window_length=slidingWindow,
                    overlap=1,
                    n_clusters=n_clusters,
                    state_size=state_size,
                    model=IFOREST_NAME,
                    tabpfn_device='cuda',
                )
                start = time.time()
                clf.process(data, labels)
                end = time.time()
                score = clf.decision_scores_
                score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
                metrics = get_metrics(score, labels)
                result = {
                    'normality_levels': normality_levels,
                    'files': json.dumps(experiment_files),
                    'series_length': series_length,
                    'nof_anomalies': nof_anomalies,
                    'method_name': modelName,
                    'tag': tag,
                    'execution_time': end - start,
                    'extra_info': json.dumps(extra_info),
                    **metrics,
                }
                insert_experiment_result(result)

            ### Online 2 - IForest with TabPFN
            modelName = LOF_NAME + "_TabPFN"
            tag = ONLINE_TAG
            if not experiment_exists(experiment_files, modelName, tag, extra_info):
                clf = TabPFNGLaSSDetector(
                    batch_frac=0.1,
                    window_length=slidingWindow,
                    overlap=1,
                    n_clusters=n_clusters,
                    state_size=state_size,
                    model=LOF_NAME,
                    tabpfn_device='cuda',
                )
                start = time.time()
                clf.process(data, labels)
                end = time.time()
                score = clf.decision_scores_
                score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
                metrics = get_metrics(score, labels)
                result = {
                    'normality_levels': normality_levels,
                    'files': json.dumps(experiment_files),
                    'series_length': series_length,
                    'nof_anomalies': nof_anomalies,
                    'method_name': modelName,
                    'tag': tag,
                    'execution_time': end - start,
                    'extra_info': json.dumps(extra_info),
                    **metrics,
                }
                insert_experiment_result(result)

