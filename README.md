# SALT: Streaming Anomaly Labeling for Time series
## Overview
SALT (Streaming Anomaly Labeling for Time Series) is a clustering-based approach for detecting anomalies in streaming time series data. SALT leverages latent space representations and probabilistic sampling to handle concept drift and real-time constraints efficiently. The method is designed to work with algorithms like Isolation Forest (IForest) and Local Outlier Factor (LOF), providing flexibility and robustness in anomaly detection.

## Key Features
- History-aware: Maintains a state of historical subsequences to capture evolving patterns.

- Concept-aware: Adapts to changes in data distribution (concept drift) using weighted sampling.

- Light-weight: Optimized for memory efficiency with an upper-bounded state size.

- Latent Space Representation: Uses PCA or TabPFN embeddings to improve separation between normal and anomalous points.

## Workflow
1. Batch Loading: Data arrives in batches of a specific length.

2. Subsequence Extraction: Extracts subsequences from each batch.

3. Embedding Extraction: Transforms subsequences into latent space (e.g., using PCA or TabPFN).

4. Clustering: Applies k-Means clustering on the embeddings.

5. Anomaly Detection: Uses IForest or LOF on each cluster to compute anomaly scores.

6. State Update: Combines current and historical subsequences via weighted sampling to update the state.

## Datasets
SALT is evaluated on two datasets:

- NAB: 58 time series, average length of 6,301 points.

- YAHOO: 367 time series, average length of 1,561 points.

