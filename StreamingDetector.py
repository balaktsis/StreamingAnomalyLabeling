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
                 state_size=None,
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
        self.state_subseq = []  # [(timestamp, subseq)]
        self.tabpfn_clf = TabPFNClassifier(n_estimators=1, random_state=self.random_state, device=tabpfn_device)
        self.tabpfn_reg = TabPFNRegressor(n_estimators=1, random_state=self.random_state, device=tabpfn_device)

    def split_batches(self, series):
        N = len(series)
        B = int(math.floor(self.batch_frac * N))
        B = min(1000, B) # todo increase limit to 10K for final experiments
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

    def z_normalize(self, embeddings):
        """Normalizes the embeddings to have zero mean and unit variance. Normalizes each row separately."""
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        normalized_embeddings = scaler.fit_transform(embeddings.T).T  # Transpose to normalize every row
        return normalized_embeddings

    def embed(self, subseqs, debug=False):
        # todo remove debug flag before running final experiments (we will need Kaggle/Colab)
        X_train = np.array(subseqs)
        y_train = X_train[:, -1] # assuming last column is the target variable
        X_train = X_train[:, :-1]
        embedding_extractor = TabPFNEmbedding(
            # tabpfn_clf=self.tabpfn_clf,
            tabpfn_reg=self.tabpfn_reg, # we only need a regressor since time series are numerical
            n_fold=0
        )
        # print(f"Extracting embeddings for {len(X_train)} subsequences with shape {X_train.shape}")
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


    def process(self, series, labels=None):
        all_scores, offset = [], 0

        print(f"Starting processing of the time series with length {len(series)} and shape {np.array(series).shape}.")

        batches = self.split_batches(series)
        labels_per_batch = self.split_batches(labels)
        slidingWindow = find_length(batches[0])

        # I had to set the sliding window length to the first batch's length
        # because if using `find_length(series)` for each batch, it might return different lengths in each batch
        # this sounds more reasonable than using the first batch's length, but with the current implementation
        # we had inconsistent lengths when concatenating with the previous state. If time we might want to debug this.
        self.window_length = slidingWindow

        for batch_idx, batch in enumerate(batches):
            # print()
            # print(f"Processing batch {batch_idx + 1}/{len(batches)} with {len(batch)} data points.")
            # print(f"Processing batch with {len(batch)} data points and shape {np.array(batch).shape}.")
            # Extract new subsequences
            subseqs, times = self.extract_subsequences(batch, t0=offset, overlap=1)
            labels_per_subsequence, _ = self.extract_subsequences(labels_per_batch[batch_idx], t0=offset, overlap=1)
            colors = ['blue' if sum(l) == 0 else 'red' for l in labels_per_subsequence]
            # print(labels_per_subsequence)
            # exit()
            # X_data = Window(window=slidingWindow).convert(batch).to_numpy()
            # print(f"Extracted {len(subseqs)} subsequences with shape {subseqs.shape}.")
            # print(f"Extracted {len(X_data)} subsequences from the batch with shape {X_data.shape}.")

            # print(f"Current state shape: {len(self.state_subseq)} x {len(self.state_subseq[0][1]) if self.state_subseq else 0}.")
            # Combine with retained state
            combined = self.state_subseq + list(zip(times.tolist(), subseqs))
            # print(f"Combined subsequences with retained state. Total subsequences: {len(combined)}.")
            combined_times = np.array([t for t, _ in combined])
            combined_seqs  = np.stack([s for _, s in combined])

            raw_data = np.concatenate((np.array(times).reshape(-1, 1), subseqs), axis=1)
            # print(raw_data[0])
            # lower_dim = PCA(n_components=2)
            lower_dim = TSNE(random_state=22, max_iter=10000, metric="cosine")
            scaler = StandardScaler()

            # Fit and transform
            raw_data_2d = scaler.fit_transform(
                lower_dim.fit_transform(scaler.fit_transform(raw_data)),
            )

            # Embed subsequences and preserve mapping to original subsequences
            embeddings = self.embed(np.concatenate((subseqs, np.array(times).reshape(-1, 1)), axis=1))
            embeddings_2d = scaler.fit_transform(
                lower_dim.fit_transform(scaler.fit_transform(embeddings)),
            )

            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.scatter(raw_data_2d[:, 0], raw_data_2d[:, 1], c=colors, label='Raw Data', alpha=0.4)
            plt.title('Raw Data Embeddings')

            plt.subplot(1, 2, 2)
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, label='Embedded Data', alpha=0.4)
            plt.title('Embedded Data')

            plt.show()

            plt.close()

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
                ages[c]  = np.median(combined_times[idx])

                # Fit anomaly detection on all subsequences in this cluster
                subseqs_cluster = np.stack([combined_seqs[i] for i in idx])
                # print(f"Shape of subsequences in cluster {c+1}: {subseqs_cluster.shape}")
                # print(f"Shape of raveled subsequences in cluster {c+1}: {subseqs_cluster.ravel().shape}")
                # exit()
                # print(f"Fitting anomaly detection on cluster {c+1} with {len(subseqs_cluster)} subsequences")

                if self.model == "iforest":
                    clf = IForest(n_jobs=1)
                    clf.fit(subseqs_cluster)
                    raw_scores = clf.detector_.decision_function(subseqs)
                elif self.model == "matrixprofile":
                    # window = find_length(subseqs_cluster)
                    # print(f"Using sliding window length: {slidingWindow}")
                    clf = MatrixProfile(window=slidingWindow)
                    clf.fit(batch)
                    raw_scores = clf.decision_scores_
                else:
                    raise ValueError(f"Unknown model: {self.model}. Supported models are 'iforest' and 'matrixprofile'.")

                # print(f"Model fitted. Attempting inference on all subsequences of the current batch.")
                # print(f"Shape of all subsequences in the batch: {subseqs.shape}")

                # Get decision scores for all the subsequences of the current batch (and not only this cluster)
                scores_for_current_batch_from_submodel = MinMaxScaler().fit_transform(
                    # using the private attribute `detector_` because `check_fitted` seems broken in TSB's model
                    # When using `clf.decision_function(subseqs)`, it raises an error that the model is not fitted
                    raw_scores.reshape(-1, 1)
                ).ravel()

                scores_for_current_batch_from_all_models.append(scores_for_current_batch_from_submodel.tolist())

            weights = {c: sizes[c] * ages[c] for c in sizes}
            sum_of_weights = sum(weights.values())

            assert(sum_of_weights > 0), "Something is wrong here. The sum of weights of all clusters is zero."
            # Normalize cluster weights so that they sum to 1
            weights = {c: w / sum_of_weights for c, w in weights.items()}

            # aggregate the scores for each subsequence. The anomaly score of each subsequence is
            # the anomaly score that each model/cluster produced for the subsequence, weighted by the cluster's weight
            agg_subseq_scores = np.zeros(subseqs.shape[0], dtype=float)
            for subseq_idx in range(len(subseqs)):
                for scores in scores_for_current_batch_from_all_models:
                    agg_subseq_scores[subseq_idx] += scores[subseq_idx] * weights[labels[subseq_idx]]
            # print(f"Anomaly score for each subsequence was successfully computed.")
            # print(f"Scores before padding: {agg_subseq_scores.shape}.")
            # print(f"Attempting to add padding to the scores if necessary.")
            sliding_window = self.window_length or find_length(batch)
            agg_subseq_scores = np.array([agg_subseq_scores[0]]*math.ceil((sliding_window-1)/2) + list(agg_subseq_scores) + [agg_subseq_scores[-1]]*((sliding_window-1)//2))
            # print(f"Scores after padding: {agg_subseq_scores.shape}.")
            all_scores.append(agg_subseq_scores)

            # Update state and offset
            self.update_state(list(zip(times.tolist(), subseqs)))
            offset += len(batch)

        return np.concatenate(all_scores)
