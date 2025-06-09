import random
import time
import json
import warnings; warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from TSB_UAD.utils.slidingWindows import find_length
from TSB_UAD.models.feature import Window
from TSB_UAD.models.iforest import IForest
from TSB_UAD.models.lof import LOF
from TSB_UAD.vus.metrics import get_metrics
from TSB_UAD.models.sand import SAND

from PCAGLaSSDetector import PCAGLaSSDetector
from experiment_generator import experiments_generator
from db_utils import experiment_exists, insert_experiment_result
from experiment_configs import IFOREST_NAME, LOF_NAME, SAND_NAME, OFFLINE_TAG, BATCH_TAG, ONLINE_TAG


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
    X_data = Window(window=slidingWindow).convert(data).to_numpy()

    ### Offline 1 - IForest
    modelName = IFOREST_NAME
    tag = OFFLINE_TAG
    if not experiment_exists(experiment_files, modelName, tag):
        clf = IForest(n_jobs=-1)
        start = time.time()
        clf.fit(X_data)
        score = clf.decision_scores_
        end = time.time()
        score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
        score = np.array(
            [score[0]] * math.ceil((slidingWindow - 1) / 2) + list(score) + [score[-1]] * ((slidingWindow - 1) // 2))
        metrics = get_metrics(score, labels)
        result = {
            'normality_levels': normality_levels,
            'files': json.dumps(experiment_files),
            'series_length': series_length,
            'nof_anomalies': nof_anomalies,
            'method_name': modelName,
            'tag': tag,
            'execution_time': end - start,
            **metrics,
        }
        insert_experiment_result(result)

    ### Offline 2 - LOF
    modelName = LOF_NAME
    tag = OFFLINE_TAG
    if not experiment_exists(experiment_files, modelName, tag):
        clf = LOF(n_neighbors=20, n_jobs=-1) # taken from https://github.com/TheDatumOrg/TSB-UAD/blob/main/example/notebooks/test_anomaly_detectors.ipynb
        start = time.time()
        clf.fit(X_data)
        score = clf.decision_scores_
        end = time.time()
        score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
        score = np.array([score[0]] * math.ceil((slidingWindow - 1) / 2) + list(score) + [score[-1]] * ((slidingWindow - 1) // 2))
        metrics = get_metrics(score, labels)
        result = {
            'normality_levels': normality_levels,
            'files': json.dumps(experiment_files),
            'series_length': series_length,
            'nof_anomalies': nof_anomalies,
            'method_name': modelName,
            'tag': tag,
            'execution_time': end - start,
            **metrics,
        }
        insert_experiment_result(result)

    ### Offline 3 - SAND
    modelName = SAND_NAME
    tag = OFFLINE_TAG
    if not experiment_exists(experiment_files, modelName, tag):
        clf = SAND(pattern_length=slidingWindow, subsequence_length=4 * (slidingWindow))
        start = time.time()
        clf.fit(data, overlaping_rate=int(1.5 * slidingWindow))
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
            **metrics,
        }
        insert_experiment_result(result)


    #### Batching
    batch_size = math.ceil(0.1 * len(data))  # 10% of the data length
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    ### Batch 1 - IForest
    modelName= IFOREST_NAME
    tag = BATCH_TAG
    if not experiment_exists(experiment_files, modelName, tag):
        slidingWindow = find_length(batches[0])
        final_scores = None
        start = time.time()
        for i, batch in enumerate(batches):
            X_data = Window(window=slidingWindow).convert(batch).to_numpy()
            clf = IForest(n_jobs=-1)
            clf.fit(X_data)
            score = clf.decision_scores_
            score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()

            scores_for_current_batch = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
            final_scores = scores_for_current_batch if i == 0 else np.concatenate((final_scores, scores_for_current_batch))

        end = time.time()
        metrics = get_metrics(final_scores, labels)
        result = {
            'normality_levels': normality_levels,
            'files': json.dumps(experiment_files),
            'series_length': series_length,
            'nof_anomalies': nof_anomalies,
            'method_name': modelName,
            'tag': tag,
            'execution_time': end - start,
            **metrics,
        }
        insert_experiment_result(result)

    ### Batch 2 - LOF
    modelName = LOF_NAME
    tag = BATCH_TAG
    if not experiment_exists(experiment_files, modelName, tag):
        slidingWindow = find_length(batches[0])
        final_scores = None
        start = time.time()
        for i, batch in enumerate(batches):
            X_data = Window(window=slidingWindow).convert(batch).to_numpy()
            clf = LOF(n_neighbors=20, n_jobs=-1)
            clf.fit(X_data)
            score = clf.decision_scores_
            score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()

            scores_for_current_batch = np.array([score[0]] * math.ceil((slidingWindow - 1) / 2) + list(score) + [score[-1]] * ((slidingWindow - 1) // 2))
            final_scores = scores_for_current_batch if i == 0 else np.concatenate(
                (final_scores, scores_for_current_batch))

        end = time.time()
        metrics = get_metrics(final_scores, labels)
        result = {
            'normality_levels': normality_levels,
            'files': json.dumps(experiment_files),
            'series_length': series_length,
            'nof_anomalies': nof_anomalies,
            'method_name': modelName,
            'tag': tag,
            'execution_time': end - start,
            **metrics,
        }
        insert_experiment_result(result)

    ### Batch 3 - SAND
    modelName = SAND_NAME
    tag = BATCH_TAG
    if not experiment_exists(experiment_files, modelName, tag):
        slidingWindow = find_length(data)
        clf = SAND(pattern_length=slidingWindow, subsequence_length=4 * (slidingWindow))
        start = time.time()
        clf.fit(data, online=True, alpha=0.5, init_length=batch_size, batch_size=batch_size, verbose=False, overlaping_rate=int(4 * slidingWindow))
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
            **metrics,
        }
        insert_experiment_result(result)

    for n_clusters in [10, 20, 40, 80]:
        for state_size in [1000, 3000, 5000]:
            for n_dimensions in [2, 3]:
                extra_info = {
                    'n_clusters': n_clusters,
                    'state_size': state_size,
                    'n_dimensions': n_dimensions,
                }

                ### Online 1 - IForest with PCA
                modelName = IFOREST_NAME + "_PCA"
                tag = ONLINE_TAG
                if not experiment_exists(experiment_files, modelName, tag, extra_info):
                    clf = PCAGLaSSDetector(
                        batch_frac=0.1,
                        window_length= slidingWindow,
                        overlap=1,
                        n_clusters=n_clusters,
                        state_size=state_size,
                        model = IFOREST_NAME,
                        n_dimensions=n_dimensions,
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

                ### Online 2 - LOF with PCA
                modelName = LOF_NAME + "_PCA"
                tag = ONLINE_TAG
                if not experiment_exists(experiment_files, modelName, tag, extra_info):
                    clf = PCAGLaSSDetector(
                        batch_frac=0.1,
                        window_length=slidingWindow,
                        overlap=1,
                        n_clusters=n_clusters,
                        state_size=state_size,
                        model=LOF_NAME,
                        n_dimensions=n_dimensions,
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

