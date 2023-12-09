import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\dmhw3\hw3M11223042\banana.csv')

df_X = df[['x','y']]
df_Y = df.iloc[:,2].values
from sklearn import cluster, datasets, metrics

banana_kmeans_fit = cluster.KMeans(n_clusters=2).fit(df_X) 

banana_cluster = banana_kmeans_fit.labels_

print("KMeans分群結果")
print(banana_cluster)
print("原本class")
print(df_Y)

import matplotlib.pyplot as plt


#KMeans
plt.subplot(2,2,1)
plt.scatter(df['x'],df['y'],c=banana_cluster)
plt.title("KMeans")
#原始分類
plt.subplot(2,2,2)
plt.scatter(df['x'], df['y'], c=df['class'])
plt.title("raw data")

plt.suptitle("exp1")
plt.show()

km = cluster.KMeans(n_clusters=2)
y_km = km.fit_predict(df_X)

plt.scatter(df_X[df_Y==0,0],
            df_X[df_Y==0,1],
            c=purple,marker='+',edgecolor='black',label='cluster1')
plt.scatter(df_X[df_Y==1,0],
            df_X[df_Y==1,1],
            c=yellow,marker='o',edgecolor='black',label='cluster2')
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.show()

