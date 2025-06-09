from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from TSB_UAD.models.matrix_profile import MatrixProfile
from TSB_UAD.utils.slidingWindows import find_length
from TSB_UAD.models.iforest import IForest
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Generalized Latent Space Streaming Detector (GLaSS Detector)
class GLaSSDetector(ABC):
    def __init__(self,
                 batch_frac=0.1,
                 window_length=None,
                 overlap=1,
                 n_clusters=10,
                 state_size=3000,
                 random_state=42,
                 max_batch_size=np.Inf,
                 max_sliding_window_length=np.Inf,
                 model=None, ):
        self.batch_frac = batch_frac
        self.window_length = window_length
        self.overlap = overlap
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.state_size = state_size
        self._max_batch_size = max_batch_size
        self._max_sliding_window_length = max_sliding_window_length
        self.model = model
        self.state_subseq = []  # [(timestamp, running_indices, subseq, labels)]
        self._decision_scores = None  # Placeholder for decision scores

    @property
    def max_batch_size(self):
        return self._max_batch_size

    @property
    def max_sliding_window_length(self):
        return self._max_sliding_window_length

    @property
    def decision_scores_(self):
        """
        Getter for decision scores.
        This is a placeholder to ensure compatibility with the TSB_UAD interface.
        """
        if self._decision_scores is None:
            raise ValueError("Decision scores have not been set yet. Call the `process` method first to compute them.")
        return self._decision_scores

    @decision_scores_.setter
    def decision_scores_(self, value):
        """
        Setter for decision scores.
        This is a placeholder to ensure compatibility with the TSB_UAD interface.
        """
        self._decision_scores_ = value

    def split_batches(self, series):
        """
        Splits the time series into batches based on the specified batch fraction, and maximum batch size.
        The maximum batch size is always respected, even if the batch fraction would suggest a larger size.
        If `state_size` is not set, it will be initialized to the batch size.

        :param series: A time series data as a list or numpy array.
        :return: A list of batches, where each batch is a segment of the time series.
        """
        N = len(series)
        B = int(math.floor(self.batch_frac * N))
        B = min(self.max_batch_size, B)
        if self.state_size is None:
            self.state_size = B
        indices = list(range(0, N, B)) + [N]
        return [series[indices[i]:indices[i + 1]] for i in range(len(indices) - 1)]

    def extract_subsequences(self, batch, t0=0, overlap=None):
        """
        Extracts subsequences from the given batch of time series data.
        The subsequences are extracted with a sliding window approach, where the window length is determined by
        `self.window_length` or the length of the batch if not set. The overlap between subsequences is determined by
        `self.overlap` or the provided `overlap` parameter.
        The function returns the subsequences, their corresponding timestamps, and running indices.
        :param batch: The batch of time series data from which to extract subsequences.
        :param t0: The initial timestamp to start counting from, useful for maintaining the time context of subsequences.
        :param overlap: The overlap between subsequences. If not provided, it defaults to `self.overlap`.
        :return: A tuple containing:
            - subsequences: A numpy array of extracted subsequences.
            - timestamps: A numpy array of timestamps corresponding to the center of each subsequence.
            - running_indices: A numpy array of indices representing the position of each subsequence in the original batch.
        """
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

    @abstractmethod
    def get_latent_representation(self, data, debug=False):
        pass

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

    def get_anomaly_scores(self, cluster_subseqs=None, batch_subseqs=None, batch=None):
        if self.model == "iforest":
            clf = IForest(n_jobs=1)
            clf.fit(cluster_subseqs)
            raw_scores = clf.detector_.decision_function(batch_subseqs)
        elif self.model == "matrixprofile":
            clf = MatrixProfile(window=self.window_length)
            clf.fit(batch)
            raw_scores = clf.decision_scores_
        else:
            raise ValueError(
                f"Unknown model: {self.model}. Supported models are 'iforest' and 'matrixprofile'.")

    def process(self, series, labels=None):
        all_scores, offset = [], 0

        print(f"Starting processing time series with length {len(series)} and shape {np.array(series).shape}.")

        batches = self.split_batches(series)
        labels_per_batch = self.split_batches(labels)
        sliding_window = min(find_length(batches[0]), self.max_sliding_window_length)
        print(f"Sliding windows are set to length {sliding_window}")

        # I had to set the sliding window length to the first batch's length
        # because if using `find_length(series)` for each batch, it might return different lengths in each batch
        # this sounds more reasonable than using the first batch's length, but with the current implementation
        # we had inconsistent lengths when concatenating with the previous state. If time we might want to debug this.
        # self.window_length = sliding_window
        # self.window_length = 20
        self.window_length = sliding_window

        for batch_idx, batch in enumerate(batches):
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
            # Embed subsequences and preserve mapping to original subsequences
            embeddings = self.get_latent_representation(self.get_enriched_subsequences(combined))

            # concatenate horizontally the running indices with the subsequence values to give TabPFN the sense of time
            # raw_data = self.get_enriched_subsequences(combined)
            # raw_data_labels = self.get_labels(combined)
            # self.plot_raw_vs_latent_space_2D(raw_data, embeddings, raw_data_labels, batch_idx)

            # Cluster with KMeans on normalized embeddings (equivalent to cosine k-means)
            # Note that `get_latent_representations()` must return already normalized representations
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

                raw_scores = self.get_anomaly_scores(subseqs_cluster, batch)

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
        # return MinMaxScaler().fit_transform(np.concatenate(all_scores).reshape(-1, 1)).ravel()
        self.decision_scores_ = MinMaxScaler().fit_transform(np.concatenate(all_scores).reshape(-1, 1)).ravel()
