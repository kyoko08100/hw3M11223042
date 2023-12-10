import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\dmhw3\hw3M11223042\banana.csv')

df_X = df[['x','y']]
raw = df.iloc[:,2].values
from sklearn import cluster, datasets, metrics

banana_kmeans_fit = cluster.KMeans(n_clusters=2).fit(df_X) 

banana_cluster = banana_kmeans_fit.labels_

print("KMeans分群結果")
print(banana_cluster)
print("原本class")
print(raw)

import matplotlib.pyplot as plt

"""
#KMeans
plt.subplot(2,2,1)
plt.scatter(df['x'],df['y'],c=banana_cluster)
plt.title("KMeans")
#原始分類
plt.subplot(2,2,2)
plt.scatter(df['x'], df['y'], c=df['class'])
plt.title("raw data")
"""
X = df.iloc[:,0:2].values
km = cluster.KMeans(n_clusters=2)
y_km = km.fit_predict(X)

#plt.subplot(2,2,3)
plt.scatter(X[y_km==0,0],
            X[y_km==0,1],
            c='purple',marker='+',edgecolor='black',label='cluster1')
plt.scatter(X[y_km==1,0],
            X[y_km==1,1],
            c='yellow',marker='o',edgecolor='black',label='cluster2')
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=300,c='red',marker='*',edgecolor='black',label='Centroids')
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
#plt.title("label")
#plt.suptitle("exp1")
plt.show()

