import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import pandas as pd
from sklearn.manifold import TSNE


#---------------------------- DBSCAN --------------------------------#
# Đọc dữ liệu
df = pd.read_csv('source/data/input/breast_cancer_dataset.csv')

# Chuyển đổi dữ liệu thành mảng NumPy
X = df.select_dtypes(include=[np.number]).values

# DBSCAN
db = DBSCAN(eps=100, min_samples=10)
y_db = db.fit_predict(X)

# Tính số lượng mỗi cụm
unique_labels, counts = np.unique(y_db, return_counts=True)
print("Number of elements in each cluster (including noise if any):")
for label, count in zip(unique_labels, counts):
    cluster_name = "Noise (-1)" if label == -1 else f"Cluster {label}"
    print(f"  {cluster_name}: {count}")

# Tạo khung
plt.figure(figsize=(12, 6))

# Biểu đồ 1: Dữ liệu gốc
plt.subplot(1, 2, 1)  # 1 hàng, 2 cột, biểu đồ đầu tiên
plt.scatter(X[:, 0], X[:, 1], c='gray', s=30)
plt.title("Original Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Biểu đồ 2: Kết quả gom nhóm DBSCAN
plt.subplot(1, 2, 2)  # 1 hàng, 2 cột, biểu đồ thứ hai
plt.scatter(X[:, 0], X[:, 1], c=y_db, cmap='tab10', s=30)
plt.title("DBSCAN")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()

# Chỉ lấy các điểm không phải noise (-1)
mask = y_db != -1
X_clustered = X[mask]
y_clustered = y_db[mask]


#---------------------------- SỬ DỤNG T-SNE --------------------------------#
# Sử dụng t-SNE để giảm chiều dữ liệu xuống 2D
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Tạo mặt nạ
mask_noise = y_db == -1
mask_clustered = ~mask_noise  # y_db != -1

# Tạo biểu đồ
plt.figure(figsize=(8, 6))

# Vẽ các điểm thuộc cụm (dùng cmap)
scatter_clustered = plt.scatter(
    X_tsne[mask_clustered, 0], X_tsne[mask_clustered, 1],
    c=y_db[mask_clustered], cmap='tab10', s=30, edgecolor='none', label='Clusters'
)

# Vẽ các điểm nhiễu riêng bằng màu xám
scatter_noise = plt.scatter(
    X_tsne[mask_noise, 0], X_tsne[mask_noise, 1],
    c='gray', s=30, edgecolor='none', label='Noise (-1)'
)

# Tiêu đề và trục
plt.title('t-SNE Visualization for DBSCAN Clustering')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')

# Lưu hình
output_path = 'source/data/output/tsne_dbscan_clustering.png'
plt.savefig(output_path)
print(f"Saved t-SNE graph at: {output_path}")

# Hiển thị
plt.show()