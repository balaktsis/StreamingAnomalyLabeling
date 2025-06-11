import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from TSB_UAD.models.matrix_profile import MatrixProfile
from TSB_UAD.utils.slidingWindows import find_length
from TSB_UAD.models.iforest import IForest
from TSB_UAD.models.feature import Window
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.embedding import TabPFNEmbedding
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class StreamingDetector:
    def __init__(self,
                 batch_frac=0.1,
                 window_length=None,
                 overlap=10,
                 n_clusters=4,
                 state_size=10000,
                 random_state=42,
                 model=None,
                 tabpfn_device='cpu'):
        self.batch_frac = batch_frac
        self.model = model
        self.window_length = window_length
        self.overlap = overlap
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.state_size = state_size
        self.state_subseq = []  # [(timestamp, running_indices, subseq, labels)]
        self.tabpfn_reg = TabPFNRegressor(n_estimators=1, random_state=self.random_state, device=tabpfn_device)

    def split_batches(self, series):
        N = len(series)
        B = int(math.floor(self.batch_frac * N))
        B = min(5000, B)  # todo increase limit to 10K for final experiments
        if self.state_size is None:
            self.state_size = B
        indices = list(range(0, N, B)) + [N]
        return [series[indices[i]:indices[i + 1]] for i in range(len(indices) - 1)]

    def extract_subsequences(self, batch, t0=0, overlap=None):
        # todo function needs docstrings; returns subsequences, timestamps and running indices
        L = self.window_length or find_length(batch)
        step = overlap or self.overlap
        batch_indices = np.arange(len(batch)) + t0
        subseqs, timestamps, running_indices = [], [], []
        for start in range(0, len(batch) - L + 1, step):
            subseqs.append(batch[start:start + L])
            running_indices.append(batch_indices[start:start + L])
            timestamps.append(t0 + start + L // 2)
        return np.array(subseqs), np.array(timestamps), np.array(running_indices)

    def z_normalize(self, embeddings):
        """Normalizes the embeddings to have zero mean and unit variance. Normalizes each row separately."""
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        normalized_embeddings = scaler.fit_transform(embeddings.T).T  # Transpose to normalize every row
        return normalized_embeddings

    def embed(self, subseqs, debug=False):
        X_train = np.array(subseqs)
        y_train = X_train[:, -1]  # assuming last column is the target variable
        X_train = X_train[:, :-1]
        embedding_extractor = TabPFNEmbedding(
            tabpfn_reg=self.tabpfn_reg,  # we only need a regressor since time series are numerical
            n_fold=0
        )

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
            )[0]
            print(f"Extracted Embeddings shape: {embeddings.shape}")

        return self.z_normalize(embeddings)

    def get_times(self, combined_data):
        return np.array([t for t, _, _, _ in combined_data], dtype=float)

    def get_running_indices(self, combined_data):
        return np.array([i for _, i, _, _ in combined_data], dtype=float)

    def get_subsequences(self, combined_data):
        return np.array([s for _, _, s, _ in combined_data], dtype=float)

    def get_labels(self, combined_data):
        return np.array([l for _, _, _, l in combined_data], dtype=float)

    def get_enriched_subsequences(self, combined_data):
        return np.concatenate(
            (self.get_running_indices(combined_data), self.get_subsequences(combined_data)),
            axis=1
        )

    def update_state(self, new_items):
        self.state_subseq.extend(new_items)

        if len(self.state_subseq) <= self.state_size:
            return

        # if len(self.state_subseq) > self.state_size:
        # Perform probabilistic sampling, giving higher weight to recent subsequences
        # times = np.array([t for t, _, _ in self.state_subseq], dtype=float)
        times = self.get_times(self.state_subseq)
        probs = times - times.min() + 1e-6  # Ensure positivity
        probs /= probs.sum()

        idx_keep = np.random.choice(
            len(self.state_subseq),
            size=self.state_size,
            replace=False,
            p=probs
        )
        self.state_subseq = [self.state_subseq[i] for i in idx_keep]

    def plot_raw_vs_latent_space_2D(self, raw_data, embeddings, labels_per_subsequence, batch_idx):
        lower_dim = PCA(n_components=2)
        # lower_dim = TSNE(random_state=22, max_iter=10000, metric="cosine")
        scaler = StandardScaler()

        # Fit and transform
        raw_data_2d = scaler.fit_transform(
            lower_dim.fit_transform(scaler.fit_transform(raw_data)),
        )

        embeddings_2d = scaler.fit_transform(
            lower_dim.fit_transform(scaler.fit_transform(embeddings)),
        )

        plt.figure(figsize=(12, 6))
        colors = ['dodgerblue' if sum(l) == 0 else 'red' for l in labels_per_subsequence]

        plt.subplot(1, 2, 1)
        plt.scatter(raw_data_2d[:, 0], raw_data_2d[:, 1], c=colors, label='Raw Data', alpha=0.4)
        plt.title('Raw Data Embeddings')

        plt.subplot(1, 2, 2)
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, label='Embedded Data', alpha=0.4)
        plt.title('Embedded Data')

        plt.savefig(f"{batch_idx}.png")

        plt.close()

    def process(self, series, labels=None):
        all_scores, offset = [], 0

        print(f"Starting processing time series with length {len(series)} and shape {np.array(series).shape}.")

        batches = self.split_batches(series)
        labels_per_batch = self.split_batches(labels)
        slidingWindow = min(find_length(batches[0]), 250)
        print(f"Sliding windows are set to length {slidingWindow}")

        # I had to set the sliding window length to the first batch's length
        # because if using `find_length(series)` for each batch, it might return different lengths in each batch
        # this sounds more reasonable than using the first batch's length, but with the current implementation
        # we had inconsistent lengths when concatenating with the previous state. If time we might want to debug this.
        # self.window_length = slidingWindow
        # self.window_length = 20
        self.window_length = slidingWindow

        for batch_idx, batch in enumerate(batches):

            print()
            print(f"Working on batch {batch_idx} with {len(batch)} elements.")

            subseqs, times, running_indices = self.extract_subsequences(batch, t0=offset, overlap=1)
            labels_per_subsequence, _, _ = self.extract_subsequences(labels_per_batch[batch_idx], t0=offset, overlap=1)

            if len(subseqs) == 0:
                # we had issues when the last batch was too small to extract subsequences.
                # hiding this for now by returning zero scores for these last points; if time we can do a better fix. todo
                all_scores.append([0] * len(batch))
                offset += len(batch)
                continue

            # data of current batch in the form (times, running_indices, subseqs) to match state representation
            data_of_current_batch = list(zip(times.tolist(), running_indices, subseqs, labels_per_subsequence))

            # Combine with retained state
            combined = self.state_subseq + data_of_current_batch

            # concatenate horizontally the running indices with the subsequence values to give TabPFN the sense of time
            raw_data = self.get_enriched_subsequences(combined)
            raw_data_labels = self.get_labels(combined)

            # Embed subsequences and preserve mapping to original subsequences
            embeddings = self.embed(self.get_enriched_subsequences(combined))
            # self.plot_raw_vs_latent_space_2D(raw_data, embeddings, raw_data_labels, batch_idx)

            # Cluster with KMeans on normalized embeddings (equivalent to cosine k-means)
            # Note that embed() returns already z-normalized embeddings
            labels = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state
            ).fit_predict(embeddings)

            # Per-cluster anomaly detection
            scores_for_current_batch_from_all_models = []
            sizes, ages = {}, {}
            for c in range(self.n_clusters):
                idx = np.where(labels == c)[0]
                sizes[c] = len(idx)
                ages[c] = np.median(self.get_times(combined)[idx])

                # Fit anomaly detection on all subsequences in this cluster
                subseqs_cluster = np.stack([self.get_subsequences(combined)[i] for i in idx])

                if self.model == "iforest":
                    clf = IForest(n_jobs=1)
                    clf.fit(subseqs_cluster)
                    raw_scores = clf.detector_.decision_function(subseqs)
                elif self.model == "matrixprofile":
                    clf = MatrixProfile(window=slidingWindow)
                    clf.fit(batch)
                    raw_scores = clf.decision_scores_
                else:
                    raise ValueError(
                        f"Unknown model: {self.model}. Supported models are 'iforest' and 'matrixprofile'.")

                # Get decision scores for all the subsequences of the current batch (and not only this cluster)
                scores_for_current_batch_from_submodel = MinMaxScaler().fit_transform(
                    # using the private attribute `detector_` because `check_fitted` seems broken in TSB's model
                    # When using `clf.decision_function(subseqs)`, it raises an error that the model is not fitted
                    raw_scores.reshape(-1, 1)
                ).ravel()

                scores_for_current_batch_from_all_models.append(scores_for_current_batch_from_submodel.tolist())

            weights = {c: sizes[c] * ages[c] for c in sizes}
            sum_of_weights = sum(weights.values())

            assert (sum_of_weights > 0), "Something is wrong here. The sum of weights of all clusters is zero."

            # Normalize cluster weights so that they sum to 1
            weights = {c: w / sum_of_weights for c, w in weights.items()}

            # aggregate the scores for each subsequence. The anomaly score of each subsequence is
            # the anomaly score that each model/cluster produced for the subsequence, weighted by the cluster's weight
            agg_subseq_scores = np.zeros(subseqs.shape[0], dtype=float)
            for subseq_idx in range(len(subseqs)):
                for scores in scores_for_current_batch_from_all_models:
                    agg_subseq_scores[subseq_idx] += scores[subseq_idx] * weights[labels[subseq_idx]]

            sliding_window = self.window_length or find_length(batch)
            agg_subseq_scores = np.array(
                [agg_subseq_scores[0]] * math.ceil((sliding_window - 1) / 2) + list(agg_subseq_scores) + [
                    agg_subseq_scores[-1]] * ((sliding_window - 1) // 2))
            all_scores.append(agg_subseq_scores)

            # Update state and offset
            self.update_state(data_of_current_batch)
            offset += len(batch)

        # Return the final scores for each point after normalizing them.
        return MinMaxScaler().fit_transform(np.concatenate(all_scores).reshape(-1, 1)).ravel()
