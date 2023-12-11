import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt
import math
import time

df = pd.read_csv('sizes3.csv')
df_x = df.iloc[:, :2]
df_y = df.iloc[:, 2]

# 使用K-means分群並計算分群時間
pre_t = time.time()
kmeans = KMeans(n_clusters = 4, n_init = 10).fit(df_x)
new_dy = kmeans.predict(df_x)
post_t = time.time()
print("K-means花費時間: ", round(post_t - pre_t, 4), "秒")

# 績效
def cluster_accuracy(predict_labels, true_labels):
    d = {}
    count = 0
    # 分群正確筆數 / 資料總筆數
    for i in range(len(true_labels)):
        # DBSCAN 噪點
        if predict_labels[i] == -1:
            continue
        if true_labels[i] not in d:
            d[true_labels[i]] = predict_labels[i]
        if d[true_labels[i]] == predict_labels[i]:
            count += 1
    return count / len(true_labels)

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

    # 帶入entropy公式計算，1000為總資料筆數
    for item in range(len(count)):
        true_log = 0 if count[item][0] == 0 else count[item][0] / sum(count[item]) * math.log(count[item][0] / sum(count[item]), 2)
        false_log = 0 if count[item][1] == 0 else count[item][1] / sum(count[item]) * math.log(count[item][1] / sum(count[item]), 2)
        entropy += sum(count[item]) / 1000 * (-( true_log + false_log ))
    return entropy
print("SSE:", kmeans.inertia_)
# print("SSE:", ac_sse(kmeans.labels_))
print("Accuracy:", cluster_accuracy(kmeans.labels_, df_y))
print("Entropy:", cluster_entropy(kmeans.labels_, df_y))

# 圖表尺寸設定
plt.rcParams['font.size'] = 14
plt.figure(figsize=(18, 10))

# 圖表顯示中文
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

# 原始資料
# plt.subplot(121)
# plt.title('原始資料')
# plt.scatter(df_x[df_y == 0].iloc[:, 0], df_x[df_y == 0].iloc[:, 1],c='red', marker="$1$")
# plt.scatter(df_x[df_y == 1].iloc[:, 0], df_x[df_y == 1].iloc[:, 1],c='green', marker="$2$")
# plt.scatter(df_x[df_y == 2].iloc[:, 0], df_x[df_y == 2].iloc[:, 1],c='blue', marker="$3$")
# plt.scatter(df_x[df_y == 3].iloc[:, 0], df_x[df_y == 3].iloc[:, 1],c='orange', marker="$4$")

# K-means結果
# plt.subplot(122)
# plt.title('K-means結果')
# plt.scatter(df_x[new_dy == 3].iloc[:, 0], df_x[new_dy == 3].iloc[:, 1],c='red', marker="$1$")
# plt.scatter(df_x[new_dy == 0].iloc[:, 0], df_x[new_dy == 0].iloc[:, 1],c='green', marker="$2$")
# plt.scatter(df_x[new_dy == 1].iloc[:, 0], df_x[new_dy == 1].iloc[:, 1],c='blue', marker="$3$")
# plt.scatter(df_x[new_dy == 2].iloc[:, 0], df_x[new_dy == 2].iloc[:, 1],c='orange', marker="$4$")

# 顯示圖表
# plt.tight_layout()
# plt.show()


# 使用階層式分群並計算分群時間
pre_t = time.time()
ac=AgglomerativeClustering(n_clusters=4)
new_dy = ac.fit_predict(df_x)
post_t = time.time()
print("階層式分群花費時間: ", round(post_t - pre_t, 4), "秒")

# 績效
def ac_sse(predict_labels):
    # 算出各群的中心
    cluster_centers = pd.DataFrame([df.loc[predict_labels == i].mean(axis=0) for i in range(ac.n_clusters)])
    
    # 計算每個資料點到該群中心的距離
    distances_squared = df.sub(cluster_centers.iloc[predict_labels].values) ** 2
    distances_squared = distances_squared.sum(axis=1)

    # 計算 SSE
    sse = distances_squared.sum()
    return sse
print("SSE:", ac_sse(new_dy))
print("Accuracy:", cluster_accuracy(ac.labels_, df_y))
print("Entropy:", cluster_entropy(ac.labels_, df_y))

# 原始資料
# plt.subplot(121)
# plt.title('原始資料')
# plt.scatter(df_x[df_y == 0].iloc[:, 0], df_x[df_y == 0].iloc[:, 1],c='red', marker="$1$")
# plt.scatter(df_x[df_y == 1].iloc[:, 0], df_x[df_y == 1].iloc[:, 1],c='green', marker="$2$")
# plt.scatter(df_x[df_y == 2].iloc[:, 0], df_x[df_y == 2].iloc[:, 1],c='blue', marker="$3$")
# plt.scatter(df_x[df_y == 3].iloc[:, 0], df_x[df_y == 3].iloc[:, 1],c='orange', marker="$4$")

# 階層式分群結果

# plt.subplot(122)
# plt.title('階層式分群結果')
# plt.scatter(df_x[new_dy == 3].iloc[:, 0], df_x[new_dy == 3].iloc[:, 1],c='green', marker="$2$")
# plt.scatter(df_x[new_dy == 0].iloc[:, 0], df_x[new_dy == 0].iloc[:, 1],c='red', marker="$1$")
# plt.scatter(df_x[new_dy == 1].iloc[:, 0], df_x[new_dy == 1].iloc[:, 1],c='orange', marker="$4$")
# plt.scatter(df_x[new_dy == 2].iloc[:, 0], df_x[new_dy == 2].iloc[:, 1],c='blue', marker="$3$")

# 顯示圖表
# plt.tight_layout()
# plt.show()




# 使用DBSCAN分群並計算分群時間
pre_t = time.time()
dbscan = DBSCAN(eps=1.2, min_samples=4).fit(df_x) # eps=1.2, ms=4 可分為4群
post_t = time.time()
print("DBSCAN分群花費時間: ", round(post_t - pre_t, 4), "秒")

# 原始資料
# plt.subplot(121)
# plt.title('原始資料')
# plt.scatter(df_x[df_y == 0].iloc[:, 0], df_x[df_y == 0].iloc[:, 1],c='red', marker="$1$")
# plt.scatter(df_x[df_y == 1].iloc[:, 0], df_x[df_y == 1].iloc[:, 1],c='green', marker="$2$")
# plt.scatter(df_x[df_y == 2].iloc[:, 0], df_x[df_y == 2].iloc[:, 1],c='blue', marker="$3$")
# plt.scatter(df_x[df_y == 3].iloc[:, 0], df_x[df_y == 3].iloc[:, 1],c='orange', marker="$4$")

# DBSCAN分群結果
# plt.subplot(122)
# plt.title('DBSCAN分群結果')
# plt.scatter(df_x.iloc[:, 0], df_x.iloc[:, 1], c=dbscan.labels_, cmap=plt.cm.Set1)

# 顯示圖表
# plt.tight_layout()
# plt.show()

# 績效
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
print("SSE:", dbscan_sse(dbscan.labels_))
print("Accuracy:", cluster_accuracy(dbscan.labels_, df_y))
print("Entropy:", cluster_entropy(dbscan.labels_, df_y))