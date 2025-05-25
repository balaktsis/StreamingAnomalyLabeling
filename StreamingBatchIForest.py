import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from TSB_UAD.utils.visualisation import plotFig
from TSB_UAD.utils.slidingWindows import find_length
from TSB_UAD.models.iforest import IForest
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.embedding import TabPFNEmbedding

import pprint


class StreamingBatchIForest:
    def __init__(self,
                 batch_frac=0.1,
                 window_length=None,
                 overlap=10,
                 n_clusters=4,
                 state_size=None,
                 random_state=42,
                 tabpfn_device='cpu'):
        self.batch_frac = batch_frac
        self.window_length = window_length
        self.overlap = overlap
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.state_size = state_size
        self.state_subseq = []  # [(timestamp, subseq)]
        self.tabpfn_clf = TabPFNClassifier(n_estimators=1, random_state=self.random_state, device=tabpfn_device)
        self.tabpfn_reg = TabPFNRegressor(n_estimators=1, random_state=self.random_state, device=tabpfn_device)

    def split_batches(self, series):
        N = len(series)
        B = int(math.floor(self.batch_frac * N))
        if self.state_size is None:
            self.state_size = B
        indices = list(range(0, N, B)) + [N]
        return [series[indices[i]:indices[i+1]] for i in range(len(indices)-1)]

    def extract_subsequences(self, batch, t0=0, overlap=None):
        L = self.window_length or find_length(batch)
        step = overlap or self.overlap
        subseqs, timestamps = [], []
        for start in range(0, len(batch)-L+1, step):
            subseqs.append(batch[start:start+L])
            timestamps.append(t0 + start + L//2)
        return np.array(subseqs), np.array(timestamps)

    def embed(self, subseqs, debug=True):
        # todo remove debug flag before running final experiments (we will need Kaggle/Colab)
        X_train = np.array(subseqs)
        y_train = X_train[:, -1] # assuming last column is the target variable
        X_train = X_train[:, :-1]
        embedding_extractor = TabPFNEmbedding(
            # tabpfn_clf=self.tabpfn_clf,
            tabpfn_reg=self.tabpfn_reg, # we only need a regressor since time series are numerical
            n_fold=0
        )
        print(f"Extracting embeddings for {len(X_train)} subsequences with shape {X_train.shape}")
        embeddings = None
        if debug:
            # Add a debug mode to make sure that the rest rationale is correct
            print(f"Debug mode is set to True. Embeddings are randomly generated.")
            np.random.seed(self.random_state)
            embeddings = np.random.randn(len(X_train), 128)
        else:
            embeddings = embedding_extractor.get_embeddings(
                X_train=X_train,
                y_train=y_train,
                X=X_train,
                data_source="train",
            )
        return embeddings


    def update_state(self, new_items):
        self.state_subseq.extend(new_items)

        if len(self.state_subseq) > self.state_size:
            # Perform probabilistic sampling, giving higher weight to recent subsequences
            times = np.array([t for t, _ in self.state_subseq], dtype=float)
            probs = times - times.min() + 1e-6  # Ensure positivity
            probs /= probs.sum()

            idx_keep = np.random.choice(
                len(self.state_subseq),
                size=self.state_size,
                replace=False,
                p=probs
            )
            self.state_subseq = [self.state_subseq[i] for i in idx_keep]


    def process(self, series):
        all_scores, offset = [], 0

        for batch in self.split_batches(series):
            # Extract new subsequences
            subseqs, times = self.extract_subsequences(batch, t0=offset)

            # Combine with retained state
            combined = self.state_subseq + list(zip(times.tolist(), subseqs))
            combined_times = np.array([t for t, _ in combined])
            combined_seqs  = np.stack([s for _, s in combined])

            # Embed subsequences and preserve mapping to original subsequences
            embeddings = self.embed(combined_seqs)
            embedding_pairs = list(zip(embeddings, combined_seqs))

            # Cluster with KMeans on normalized embeddings (equivalent to cosine k-means)
            labels = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state
            ).fit_predict(embeddings)

            # Per-cluster IForest
            cluster_scores = []
            sizes, ages = {}, {}
            for c in range(self.n_clusters):
                idx = np.where(labels == c)[0]
                sizes[c] = len(idx)
                ages[c]  = np.median(combined_times[idx])

                # Fit IForest on all subsequences in this cluster
                subseqs_cluster = np.stack([embedding_pairs[i][1] for i in idx])
                print(f"Fitting IForest on cluster {c} with {len(subseqs_cluster)} subsequences")

                clf = IForest(n_jobs=1)
                clf.fit(subseqs_cluster)
                print(f"IForest fitted. Attempting inference on all subsequences of the current batch.")
                print(f"Shape of all subsequences in the batch: {subseqs.shape}")

                # Get decision scores for all the subsequences of the current batch (and not only this cluster)
                sc_all = MinMaxScaler().fit_transform(
                    # using the private attribute `detector_` because `check_fitted` seems broken in TSB's IForest
                    # When using `clf.decision_function(subseqs)`, it raises an error that the model is not fitted
                    clf.detector_.decision_function(subseqs).reshape(-1, 1)
                ).ravel()

                cluster_scores.append(sc_all.tolist())

            weights = {c: sizes[c] * ages[c] for c in sizes}
            sum_of_weights = sum(weights.values())

            assert(sum_of_weights > 0), "Something is wrong here. The sum of weights of all clusters is zero."
            # Normalize cluster weights so that they sum to 1
            weights = {c: w / sum_of_weights for c, w in weights.items()}

            # aggregate the scores for each subsequence. The anomaly score of each subsequence is
            # the anomaly score that each model/cluster produced for the subsequence, weighted by the cluster's weight
            agg_subseq_scores = np.zeros(subseqs.shape[0], dtype=float)
            for subseq_idx in range(len(subseqs)):
                for scores in cluster_scores:
                    agg_subseq_scores[subseq_idx] += scores[subseq_idx] * weights[labels[subseq_idx]]

            print(f"Anomaly score for each subsequence was successfully computed.")
            pprint.pp(agg_subseq_scores)
            pprint.pp(agg_subseq_scores.shape)
            exit()
            # Unsure about the padding logic below so I stop here for now
            # `agg_subseq_scores` is an array with shape (len(subseqs),) containing the anomaly scores for each subsequence

            
            # TODO: fix the following to adapt to the modified scores length
            # WHY: len(sc_all) is now batsch_size - window_length + 1
            # HOW: pad the last scores with zeros or the last score
             
            # Map subseq scores back to the batch's time points
            L = self.window_length or find_length(batch)
            point_scores = np.zeros(len(batch))
            weights = {c: sizes[c] * ages[c] for c in sizes}
            weight_sums = np.zeros(len(batch))

            base = len(self.state_subseq)  # new subsequences start from this index

            for j, t in enumerate(times):
                c = labels[base + j]
                start = max(0, int((t - offset) - L // 2))
                end = min(len(batch), start + L)
                weight = weights[c]

                # Spread the weighted score over the data points in the subsequence
                point_scores[start:end] += cluster_scores[base + j] * weight
                weight_sums[start:end] += weight

            # Normalize by accumulated weight per point to get final anomaly score per point
            point_scores = np.divide(
                point_scores,
                weight_sums,
                out=np.zeros_like(point_scores),
                where=weight_sums != 0
            )

            # Update state and offset
            self.update_state(list(zip(times.tolist(), subseqs)))
            offset += len(batch)

        return np.concatenate(all_scores)
