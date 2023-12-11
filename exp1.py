import pandas as pd
import numpy as np
import math
from collections import Counter
from sklearn.metrics import pairwise_distances_argmin_min
#載入檔案
df = pd.read_csv(r'C:\dmhw3\hw3M11223042\banana.csv')
df_X = df[['x','y']]
raw = df.iloc[:,2].values
y=df.iloc[:,2]

from sklearn import cluster, datasets, metrics
#KMEANS
banana_kmeans_fit = cluster.KMeans(n_clusters=2).fit(df_X) 
#值
banana_cluster = banana_kmeans_fit.labels_

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
#原本class1、2轉成0、1
df['class'] = labelencoder.fit_transform(df['class'])
#0、1數量
print("原始0、1數量")
print(Counter(df['class']))
print("###############")
print(Counter(banana_cluster))
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
from sklearn.cluster import KMeans
import time
#執行時間 
KMstart = time.time()
#取X,Y值 再用km、fit、predict
x1= df.iloc[:,:2]
X = df.iloc[:,0:2].values
km = cluster.KMeans(n_clusters=2)
km1 = KMeans(n_clusters=2).fit(x1)
y_km = km.fit_predict(X)
#SSE
distortion = []
distortion.append(km.inertia_)
print("KMEANS-SSE:",distortion)
#前減後
KMend = time.time()
print("KMEANS執行時間:%f秒"%(KMend - KMstart))
#plt.subplot(2,2,3)

#類別為0印+ 為1印o 中心點印*
plt.subplot(2,2,1)
plt.title('kmeans')
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


#算ACC
from sklearn.metrics import accuracy_score 

bananaacc = accuracy_score(df['class'].values,y_km)
print("KMEANS-Accuracy:",bananaacc)
#算熵
"""
分隔線123
"""
def cluster_entropy(predict_labels, true_labels):
    d = {}
    count = {}
    entropy = 0
    # 算出每群正確及錯誤各有幾筆
    for i in range(len(true_labels)):
        if true_labels[i] not in d:
            d[true_labels[i]] = predict_labels[i]
            count[true_labels[i]] = list([0, 0])
        if d[true_labels[i]] == predict_labels[i]:
            count[true_labels[i]][0] += 1
        else:
            count[true_labels[i]][1] += 1

    # 帶入entropy公式計算，1000為總資料筆數 4811
    for item in range(len(count)):
        true_log = 0 if count[item+1][0] == 0 else count[item+1][0] / sum(count[item+1]) * math.log(count[item+1][0] / sum(count[item+1]), 2)
        false_log = 0 if count[item+1][1] == 0 else count[item+1][1] / sum(count[item+1]) * math.log(count[item+1][1] / sum(count[item+1]), 2)
        entropy += sum(count[item+1]) / 4811 * (-( true_log + false_log ))
    return entropy
    
print("KMEANS-Entropy",cluster_entropy(km1.labels_,y))
# from scipy.stats import entropy
# #算類別機率
# bananaprob = np.bincount(y_km) / len(y_km)
# #代入 以2為基底
# bana_entropy = entropy(bananaprob, base=2)
# print("KMEANS-Entropy:", bana_entropy)

#階層式
histart = time.time()
from sklearn.cluster import AgglomerativeClustering

hier = AgglomerativeClustering(n_clusters=4)
hier1 = hier.fit(x1)
bahier = hier.fit_predict(X)
hiend = time.time()
print("階層式分群執行時間:%f"%(hiend-histart))



import scipy.cluster.hierarchy as sch
dis=sch.linkage(X,metric='euclidean',method='ward')
#plt.subplot(2,2,3)
#sch.dendrogram(dis)
#plt.title('Hier')
#plt.show()
k=2
clusters =sch.fcluster(dis,k,criterion='maxclust')

# 算出各群的中心
cluster_centers = pd.DataFrame([df.loc[bahier == i].mean(axis=0) for i in range(hier.n_clusters)])
    
# 計算每個資料點到該群中心的距離
distances_squared = df.sub(cluster_centers.iloc[bahier].values) ** 2
distances_squared = distances_squared.sum(axis=1)

# 計算 SSE
sse = distances_squared.sum()
print("階層式分群SSE:",sse)
#計算ACC
hieracc = accuracy_score(df['class'].values,bahier)
print("階層式分群Accuracy:",hieracc)
#計算ENT
print("階層式分群Entropy",cluster_entropy(hier1.labels_,y))
# plt.subplot(2,2,2)
# plt.title('Hier')
# plt.scatter(df['x'],df['y'],c=clusters)

# plt.scatter(X[clusters==1,0],
#             X[clusters==1,1],
#             c='purple',marker='+',label='cluster2')
# plt.scatter(X[clusters==2,0],
#             X[clusters==2,1],
#             c='yellow',marker='o',label='cluster1')

plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.show()


#DBSCAN

from sklearn.cluster import DBSCAN
dbstart = time.time()
dbs1 = DBSCAN(eps=0.08,min_samples=50).fit(X)
dbs1lb=DBSCAN(eps=0.08,min_samples=50).fit_predict(X)
badb1 = dbs1.labels_
dbend = time.time()
print("DBSCAN執行時間:%f"%(dbend-dbstart))
#badb = dbs1.fit_predict(X)
#---test---
# print('badb',badb)
# plt.subplot(2,2,1)
# plt.title('dbs1')
# plt.scatter(df['x'],df['y'],c=badb)
plt.subplot(2,2,1)
plt.title('dbs1#eps0.08,ms50')
plt.scatter(X[badb1==0,0],
            X[badb1==0,1],
            c='purple',marker='+',label='cluster1')
plt.scatter(X[badb1==1,0],
            X[badb1==1,1],
            c='yellow',marker='o',label='cluster2')
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()

dbs2 = DBSCAN(eps=0.05,min_samples=30).fit(X)
dbs2lb=DBSCAN(eps=0.05,min_samples=30).fit_predict(X)
badb2 = dbs2.labels_
plt.subplot(2,2,2)
plt.title('dbs2#eps0.05,ms30')
plt.scatter(X[badb2==0,0],
            X[badb2==0,1],
            c='purple',marker='+',label='cluster1')
plt.scatter(X[badb2==1,0],
            X[badb2==1,1],
            c='yellow',marker='o',label='cluster2')
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
# plt.subplot(2,2,2)
# plt.title('dbs2')
# plt.scatter(X[clusters==2,0],
#             X[clusters==2,1],
#             c='purple',marker='+',label='cluster1')
# plt.scatter(X[clusters==1,0],
#             X[clusters==1,1],
#             c='yellow',marker='o',label='cluster2')
def dbscan_sse(predict_labels):
    # 算出各群的中心
    cluster_centers_idx = pairwise_distances_argmin_min(df, df[predict_labels == 0].mean().values.reshape(1, -1))[0]
    cluster_centers = df.iloc[cluster_centers_idx]

    # 計算每個資料點到該群中心的距離
    distances_squared = df.sub(cluster_centers.iloc[predict_labels].values) ** 2
    distances_squared = distances_squared.sum(axis=1)

    # 計算 SSE
    sse = distances_squared.sum()
    return sse
print("DBSCAN1-SSE", dbscan_sse(badb1))
print("DBSCAN2-SSE", dbscan_sse(badb2))
plt.show()

#算ACC
DBS1acc = accuracy_score(df['class'].values,dbs1lb)
print("DBS1-Accuracy:",DBS1acc)
DBS2acc = accuracy_score(df['class'].values,dbs2lb)
print("DBS2-Accuracy:",DBS2acc)

#計算ENT
print("DBS1-Entropy",cluster_entropy(badb1,y))
print("DBS2-Entropy",cluster_entropy(badb2,y))