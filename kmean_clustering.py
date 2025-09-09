import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE


#---------------------------- HÀM ĐỘ ĐO --------------------------------#
# Khoảng cách Euclidean
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

# Khoảng cách Manhattan (L1)
def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))

# Khoảng cách Minkowski (được điều chỉnh bởi p)
def minkowski_distance(a, b, p=3):
    return np.power(np.sum(np.abs(a - b) ** p), 1/p)


#---------------------------- HÀM KHỞI TẠO BẰNG KMEAN++ --------------------------------#
# Khởi tạo centroid theo K-means++
def kmeans_plus_plus(X, k):
    centroids = []
    # Chọn centroid đầu tiên ngẫu nhiên
    centroids.append(X.iloc[np.random.choice(X.shape[0])].values)
    
    # Chọn các centroid tiếp theo
    for _ in range(1, k):
        # Tính khoảng cách từ mỗi điểm đến centroid gần nhất
        dist = np.min([np.linalg.norm(X.values - c, axis=1) for c in centroids], axis=0)
        # Chọn điểm có xác suất lớn nhất
        prob = dist / dist.sum()  # Xác suất theo khoảng cách
        cumulative_prob = np.cumsum(prob)
        r = np.random.rand()
        idx = np.searchsorted(cumulative_prob, r)
        
        # Thêm centroid mới
        centroids.append(X.iloc[idx].values)
    
    return np.array(centroids)


#---------------------------- K-MEANS - CLUSTERING --------------------------------#
# K-means Clustering
# Với 'use_kmeans_plus_plus' là false thì áp dụng K-means Clustering, true thì áp dụng K-means++
def kmeans(X, k, distance_func=None, tolerance=1e-4, max_iters=300, use_kmeans_plus_plus=False):
    if distance_func is None:
        distance_func = euclidean_distance

    # Tạo ngẫu nhiên các centroid
    np.random.seed(42)
    if use_kmeans_plus_plus:
        centroids = kmeans_plus_plus(X, k)  # Sử dụng K-means++
    else:
        random_idx = np.random.permutation(X.shape[0])[:k]
        centroids = X.iloc[random_idx].values

    for iteration in range(max_iters):
        # Gán nhãn cụm gần nhất
        labels = np.array([
            np.argmin([distance_func(x, c) for c in centroids])
            for x in X.values
        ])

        # Tính lại tâm cụm
        new_centroids = np.array([
            X.values[labels == i].mean(axis=0)
            for i in range(k)
        ])

        # Kiểm tra hội tụ
        if np.allclose(centroids, new_centroids, atol=tolerance):
            print(f'Converged at iteration {iteration}')
            break
        centroids = new_centroids

    return centroids, labels


#---------------------------- VẼ BIỂU ĐỒ --------------------------------#
# Vẽ biểu đồ Scatter
def plot_kmeans_2d(X, labels, centroids, k, title):
    plt.figure(figsize=(6, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, k))

    for i in range(k):
        cluster_points = X.values[labels == i, :2]  # chỉ dùng 2 features đầu
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i+1}', color=colors[i])

    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='.', label='Centroids')
    plt.xlabel(f'Feature 0 ({X.columns[0]})')
    plt.ylabel(f'Feature 1 ({X.columns[1]})')
    plt.title(title)

# Đọc file CSV
df = pd.read_csv('source/data/input/breast_cancer_dataset.csv')

# Loại bỏ cột target nếu có
if 'target' in df.columns:
    X = df.drop(columns=['target'])
else:
    X = df.copy()

# Gán số cụm
k = 2

# Clustering
centroids, labels = kmeans(X, k=k, distance_func=manhattan_distance, use_kmeans_plus_plus=False)

# Số lượng mỗi cụm
unique_labels, counts = np.unique(labels, return_counts=True)
print("Number of elements in each cluster (K-means): ")
for label, count in zip(unique_labels, counts):
    print(f"  Cluster {label + 1}: {count}")

# Plot K-means for Euclidean, Manhattan, and Minkowski
plot_kmeans_2d(X, labels, centroids, k, 'K-means (Manhattan Distance)')

# Lưu biểu đồ vào file
output_path = 'source/data/output/kmeans_2d_plot.png'
plt.savefig(output_path)
print(f"The 2D chart has been saved at: {output_path}")


#---------------------------- BIỂU ĐỒ 3D --------------------------------#
# Vẽ 3D nếu >= 3 features
if X.shape[1] >= 3:
    # Vẽ kết quả 3D (lấy 3 feature đầu tiên)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Sử dụng bảng màu 'tab10'
    cmap = plt.get_cmap('tab10')

    for i in range(k):
        cluster_points = X.values[labels == i, :3]  # chỉ lấy 3 feature đầu
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], 
                   label=f'Cluster {i+1}', color=cmap(i))

    # Vẽ centroid
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], s=300, c='red', marker='.', label='Centroids')
    ax.set_xlabel(f'Feature 0 ({X.columns[0]})')
    ax.set_ylabel(f'Feature 1 ({X.columns[1]})')
    ax.set_zlabel(f'Feature 2 ({X.columns[2]})')
    ax.set_title('K-means Clustering (3D)')
    ax.legend()

plt.show()


#---------------------------- SỬ DỤNG T-SNE --------------------------------#
# Giảm chiều X xuống 2D bằng t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(X)

# Không cần t-SNE cho centroids!
# Tính lại tọa độ centroid từ X_tsne và labels
centroids_tsne = np.array([
    X_tsne[labels == i].mean(axis=0)
    for i in range(k)
])

# Vẽ scatter
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='tab10', s=30)
plt.scatter(centroids_tsne[:, 0], centroids_tsne[:, 1], c='red', marker='X', s=300, label='Centroids')
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.title('K-means Clustering Visualized by t-SNE')
plt.legend()
plt.show()