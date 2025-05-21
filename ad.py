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
from StreamingBatchIForest import StreamingBatchIForest

public_root = "./data/"
selected_domains = ['NASA-MSL', 'IOPS', 'Genesis', 'YAHOO']


### Load the data
# For each domain, pick the most anomalous ts
series_list, label_list = [], []
for dom in selected_domains:
    dom_path = os.path.join(public_root, dom)
    files = [f for f in os.listdir(dom_path) if f.endswith('.out')]
    
    max_ones = -1
    chosen = None

    for f in files:
        file_path = os.path.join(dom_path, f)
        df = pd.read_csv(file_path, header=None).dropna()
        ones_count = (df.iloc[:, 1] == 1).sum()

        if ones_count > max_ones:
            max_ones = ones_count
            chosen = f

    if chosen:
        df = pd.read_csv(os.path.join(dom_path, chosen), header=None).dropna().to_numpy()
        data = df[:, 0].astype(float)
        label = df[:, 1].astype(int)
        series_list.append(data)
        label_list.append(label)

norm1_ts = series_list[0]
norm1_labels = label_list[0]
norm2_ts = np.concatenate([series_list[0],series_list[1]], axis=0)
norm2_labels = np.concatenate([label_list[0],label_list[1]], axis=0)
norm3_ts = np.concatenate([series_list[0],series_list[1],series_list[2]], axis=0)
norm3_labels = np.concatenate([label_list[0],label_list[1],label_list[2]], axis=0)
norm4_ts = np.concatenate(series_list, axis=0)
norm4_labels = np.concatenate(label_list, axis=0)

series = [norm1_ts, norm2_ts, norm3_ts, norm4_ts]
labels = [norm1_labels, norm2_labels, norm3_labels, norm4_labels]


### Offline 1 - IForest
modelName='IForest'
norm = 2
data = series[norm-1]
label = labels[norm-1]

slidingWindow = find_length(data)
X_data = Window(window = slidingWindow).convert(data).to_numpy()

clf = IForest(n_jobs=1)
clf.fit(X_data)
score = clf.decision_scores_
score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))

plotFig(data, label, score, slidingWindow, fileName="Norm-"+str(norm)+"-"+modelName, modelName=modelName)
plt.savefig("Norm-"+str(norm)+"-"+modelName, dpi=300, bbox_inches='tight')


### Offline 2 - MatrixProfile
modelName='MatrixProfile'
norm = 2
data = series[norm-1]
label = labels[norm-1]

slidingWindow = find_length(data)
X_data = data

clf = MatrixProfile(window = slidingWindow)
clf.fit(X_data)
score = clf.decision_scores_
score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))

plotFig(data, label, score, slidingWindow, fileName="Norm-"+str(norm)+"-"+modelName, modelName=modelName)
plt.savefig("Norm-"+str(norm)+"-"+modelName, dpi=300, bbox_inches='tight')


### Offline 3 - SAND
modelName='SAND (offline)'
norm = 2
data = series[norm-1]
label = labels[norm-1]

slidingWindow = find_length(data)
X_data = data

clf = SAND(pattern_length=slidingWindow,subsequence_length=4*(slidingWindow))
clf.fit(X_data,overlaping_rate=int(1.5*slidingWindow))
score = clf.decision_scores_
score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()

plotFig(data, label, score, slidingWindow, fileName="Norm-"+str(norm)+"-"+modelName, modelName=modelName)
plt.savefig("Norm-"+str(norm)+"-"+modelName, dpi=300, bbox_inches='tight')


### Online 1 - SAND
modelName='SAND (online)'
norm = 2
data = series[norm-1]
label = labels[norm-1]

slidingWindow = find_length(data)
X_data = data

clf = SAND(pattern_length=slidingWindow,subsequence_length=4*(slidingWindow))
clf.fit(X_data,online=True,alpha=0.5,init_length=5000,batch_size=2000,verbose=True,overlaping_rate=int(4*slidingWindow))
score = clf.decision_scores_
score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
plotFig(data, label, score, slidingWindow, fileName="Norm-"+str(norm)+"-"+modelName, modelName=modelName)
plt.savefig("Norm-"+str(norm)+"-"+modelName, dpi=300, bbox_inches='tight')


### Online 2 - IForest
sb_if = StreamingBatchIForest(
    batch_frac=0.1,
    overlap=10,
    n_clusters=4,
    state_size=None,       
    tabpfn_device='cpu'
)
scores = sb_if.process(data)

sliding = find_length(data)
plotFig(data, label, scores, sliding,
        fileName='Stream_IForest', modelName='StreamIForest')
plt.savefig('Stream_IForest.png', dpi=300, bbox_inches='tight')
