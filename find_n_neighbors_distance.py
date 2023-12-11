import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
"""
df = pd.read_csv('sizes3.csv')
df_x = df.iloc[:, :2]
df_y = df.iloc[:, 2]
# 設定要找的鄰居數目
n_neighbors = 4

# 使用 NearestNeighbors 找到每個點的鄰居
neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1)  # 加1是為了包括自己
neighbors.fit(df_x)

# distances 是每個點到其鄰居的距離
# indices 是每個點的鄰居的索引
distances, indices = neighbors.kneighbors(df_x)

# 取第四個鄰居的距離
n_neighbor_distances = distances[:, n_neighbors]
n_neighbor_distances.sort()

plt.plot(range(1, len(n_neighbor_distances) + 1), n_neighbor_distances)
plt.xlabel('Point sorted according to distance of 4th nearest neighbor')
plt.ylabel('4th nearest neighbor distance')
plt.title('4th Neighbor Distances for Each Data Point')
plt.show()
"""

#banana
df = pd.read_csv('banana.csv')
df_x = df.iloc[:, :2]
df_y = df.iloc[:, 2]
# 設定要找的鄰居數目
n_neighbors = 2000

# 使用 NearestNeighbors 找到每個點的鄰居
neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1)  # 加1是為了包括自己
neighbors.fit(df_x)

# distances 是每個點到其鄰居的距離
# indices 是每個點的鄰居的索引
distances, indices = neighbors.kneighbors(df_x)

# 取第四個鄰居的距離
n_neighbor_distances = distances[:, n_neighbors]
n_neighbor_distances.sort()

plt.plot(range(1, len(n_neighbor_distances) + 1), n_neighbor_distances)
plt.xlabel('Point sorted according to distance of 4th nearest neighbor')
plt.ylabel('4th nearest neighbor distance')
plt.title('4th Neighbor Distances for Each Data Point')
plt.show()
