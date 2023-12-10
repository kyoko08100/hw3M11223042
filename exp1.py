import pandas as pd
import numpy as np
#載入檔案
df = pd.read_csv(r'C:\dmhw3\hw3M11223042\banana.csv')

df_X = df[['x','y']]
raw = df.iloc[:,2].values
from sklearn import cluster, datasets, metrics
#KMEANS
banana_kmeans_fit = cluster.KMeans(n_clusters=2).fit(df_X) 
#值
banana_cluster = banana_kmeans_fit.labels_

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
#原本class1、2轉成0、1
df['class'] = labelencoder.fit_transform(df['class'])

print("KMeans分群結果")
print(banana_cluster)
print("原本class")
print(df['class'].values)

import matplotlib.pyplot as plt

"""
#簡單比較圖
#KMeans
plt.subplot(2,2,1)
plt.scatter(df['x'],df['y'],c=banana_cluster)
plt.title("KMeans")
#原始分類
plt.subplot(2,2,2)
plt.scatter(df['x'], df['y'], c=df['class'])
plt.title("raw data")
"""

import time
#執行時間 
KMstart = time.time()
#取X,Y值 再用km、fit、predict
X = df.iloc[:,0:2].values
km = cluster.KMeans(n_clusters=2)
y_km = km.fit_predict(X)
#SSE
distortion = []
distortion.append(km.inertia_)
print("SSE:",distortion)
#前減後
KMend = time.time()
print("KMeans執行時間:%f秒"%(KMend - KMstart))
#plt.subplot(2,2,3)

#類別為0印+ 為1印o 中心點印*
plt.scatter(X[y_km==0,0],
            X[y_km==0,1],
            c='purple',marker='+',label='cluster1')
plt.scatter(X[y_km==1,0],
            X[y_km==1,1],
            c='yellow',marker='o',label='cluster2')
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=300,c='red',marker='*',label='Centroids')
#示例圖
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
#plt.title("label")
#plt.suptitle("exp1")
plt.show()
#算SSE


#算ACC
from sklearn.metrics import accuracy_score 

bananaacc = accuracy_score(df['class'].values,y_km)
print("Accuracy:",bananaacc)
#算熵
from scipy.stats import entropy
#算類別機率
bananaprob = np.bincount(y_km) / len(y_km)
#代入 以2為基底
bana_entropy = entropy(bananaprob, base=2)
print("Entropy:", bana_entropy)