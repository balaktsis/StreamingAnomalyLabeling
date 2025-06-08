import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from TSB_UAD.utils.visualisation import plotFig
from TSB_UAD.utils.slidingWindows import find_length
from TSB_UAD.models.feature import Window
from TSB_UAD.models.iforest import IForest
from TSB_UAD.models.sand import SAND
from TSB_UAD.models.matrix_profile import MatrixProfile
from StreamingDetector import StreamingDetector

# Load all time series from the YAHOO domain
public_root = "./data/"
selected_domain = 'NAB'
series_list, label_list = [], []
dom_path = os.path.join(public_root, selected_domain)
files = sorted(f for f in os.listdir(dom_path) if f.endswith('.out'))  # Ensure consistent ordering

# Keep only the first i files
i=4
for f in files:
    file_path = os.path.join(dom_path, f)
    df = pd.read_csv(file_path, header=None).dropna()
    data = df.iloc[:, 0].astype(float).to_numpy()
    label = df.iloc[:, 1].astype(int).to_numpy()
    series_list.append(data)
    label_list.append(label)
    i-=1
    if i == 0:  
        break

# Create cumulative time series and labels
series = []
labels = []
for i in range(1, len(series_list) + 1):
    cum_ts = np.concatenate(series_list[:i], axis=0)
    cum_labels = np.concatenate(label_list[:i], axis=0)
    series.append(cum_ts)
    labels.append(cum_labels)


# Create different-normalities time series
norm1_ts = series[0]
norm1_labels = labels[0]
norm2_ts = series[1] if len(series) > 1 else None
norm2_labels = labels[1] if len(labels) > 1 else None
norm3_ts = series[2] if len(series) > 2 else None
norm3_labels = labels[2] if len(labels) > 2 else None
norm4_ts = series[3] if len(series) > 3 else None
norm4_labels = labels[3] if len(labels) > 3 else None

series = [norm1_ts, norm2_ts, norm3_ts, norm4_ts]
labels = [norm1_labels, norm2_labels, norm3_labels, norm4_labels]

# Select normality norm
norm = 3
data = series[norm-1]
label = labels[norm-1]

## Variant 1 - Batching
batch_size = 500
batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

sb_mp = StreamingDetector(
batch_frac=0.1,
    overlap=10,
    n_clusters=4,
    model="iforest",
    state_size=None,
    tabpfn_device='cpu'
)
scores = sb_mp.process(data, label)
sliding = find_length(data)
plotFig(data, label, scores, sliding, fileName='Stream_IForest', modelName='StreamIForest')
plt.show()