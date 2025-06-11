![image](https://github.com/user-attachments/assets/8d6e679c-def1-4f12-9e80-bafa142c4770)

## Overview
SALT (***S***treaming ***A***nomaly ***L***abeling for ***T***ime-series) is a clustering-based approach for detecting anomalies in streaming time-series data. SALT leverages latent space representations and probabilistic sampling to handle concept drift and real-time constraints efficiently. The method is designed to work with algorithms like Isolation Forest (IForest) and Local Outlier Factor (LOF), providing flexibility and robustness in anomaly detection. The following figure illustrates an overview of SALT, which is then detailed in the [Workflow](#workflow) section below.

![image](https://github.com/user-attachments/assets/37d2db07-e86b-4aa8-a645-888ce6b0b8b5)

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

## Why Latent Space?

Our experimental investigation reveals that, in several domains, projecting the raw data points to a latent space leads to an improved separability between the normal and the anomalous time-series subsequences. Two indicative examples are shown in the following figure. The blue circles correspond to the representations of normal subsequences, while the red circles to the representations of the anomalous ones. Using an encoder (PCA or TabPFN) can lead to a more separable latent space, where anomalous points tend to form homegeneous clusters, well-separated from the clusters of normal points. We therefore use well-established time series anomaly detection techniques on the latent representations, significantly boosting their performance, as shown in the [evaluation](#datasets-and-evaluation) section below. 

![image](https://github.com/user-attachments/assets/bf5bdbb2-bc62-41e4-a28a-a6b74910d50e)


## Datasets and Evaluation
SALT is evaluated on two domains:

- NAB: 58 time series, average length of 6,301 points.

- YAHOO: 367 time series, average length of 1,561 points.

Our extensive experimental evaluation comprising more than 1'300 experimental runs on 4 notions of normality, 2 concept drift types and 2 traditional anomaly detection algorithms reveals that applying SALT yields significantly better results than applying the underlying technique in batches of data. In the following figures, we call the latter option a ``naive batching'' approach. All points correspond to experimental runs with a specific normality notion and concept drift type. Points above the main diagonal indicate that SALT outperforms the naive approach, offering significant performance boosts.

![comparison_quality_improvement](https://github.com/user-attachments/assets/5cfdbfea-76b1-4233-80ab-ebb04e5f4ab1)


## More Information
For more information about SALT, please refer to our [presentation slides](https://github.com/balaktsis/StreamingAnomalyLabeling/blob/main/salt.pdf). 
