import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import TextBox, Button, RadioButtons
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
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


#---------------------------- LIÊN KẾT TÍNH TOÁN --------------------------------#
# Hàm tính khoảng cách giữa các cụm
def calculate_cluster_distance(cluster_a, cluster_b, X, method='single'):
    # Lấy các điểm trong hai cụm
    points_a = X[cluster_a]
    points_b = X[cluster_b]
    
    if method == 'single':
        # Single linkage
        distances = [euclidean_distance(p1, p2) for p1 in points_a for p2 in points_b]
        return np.min(distances)
    elif method == 'complete':
        # Complete linkage
        distances = [euclidean_distance(p1, p2) for p1 in points_a for p2 in points_b]
        return np.max(distances)
    elif method == 'average':
        # Average linkage
        distances = [euclidean_distance(p1, p2) for p1 in points_a for p2 in points_b]
        return np.mean(distances)
    elif method == 'ward':
        # Ward linkage
        mean_a = np.mean(X[cluster_a], axis=0)
        mean_b = np.mean(X[cluster_b], axis=0)

        var_a = np.sum((X[cluster_a] - mean_a) ** 2)
        var_b = np.sum((X[cluster_b] - mean_b) ** 2)
        
        return var_a + var_b
    else:
        raise ValueError("Unknown linkage method: use 'single', 'complete', 'average', or 'ward'.")
    
#---------------------------- TẠO CÂY PHÂN CẤP --------------------------------#
# Hierarchical Clustering
def hierarchical_clustering(data, method='ward'):
    X = data.values
    linked = linkage(X, method=method)
    return linked


#---------------------------- CẮT CÂY --------------------------------#
# Cắt cây
def cut_dendrogram(linked, n_clusters=None, distance_threshold=None):
    if n_clusters is not None:
        clusters = fcluster(linked, n_clusters, criterion='maxclust')   # cắt theo số cụm
    elif distance_threshold is not None:
        clusters = fcluster(linked, distance_threshold, criterion='distance')   # cắt theo khoảng cách (distance)
    else:
        raise ValueError("Either n_clusters or distance_threshold must be specified.")
    return clusters

#---------------------------- HIỂN THỊ DỮ LIỆU GOM NHÓM --------------------------------#
# Đọc dữ liệu
df = pd.read_csv('data/input/breast_cancer_dataset.csv')

# Clustering
linked = hierarchical_clustering(df.iloc[:, :2], method='ward')

# Tạo layout figure có 3 cột (dendro - scatter - widget)
fig = plt.figure(figsize=(14, 6))
gs = gridspec.GridSpec(1, 3, width_ratios=[3, 3, 1])  # cột thứ 3 cho widget

ax_dendro = fig.add_subplot(gs[0])
ax_scatter = fig.add_subplot(gs[1])

# Vẽ dendrogram
dendro = dendrogram(linked, ax=ax_dendro, no_labels=True)
ax_dendro.set_title('Hierarchical Clustering Dendrogram')
ax_dendro.set_ylabel('Distance')

# Vẽ scatter
scatter = ax_scatter.scatter(
    df.iloc[:, 0], df.iloc[:, 1],
    c=np.zeros(len(df)), cmap='gray', edgecolor='none'
)
ax_scatter.set_title('Clusters (cutting tree)')
ax_scatter.set_xlabel('Feature 1')
ax_scatter.set_ylabel('Feature 2')

# Vẽ đường cắt
cut_line = ax_dendro.axhline(y=0, color='red', linestyle='--')

# Bảng màu
colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'gold', 'lime', 'pink']

# Widget
text_ax = fig.add_subplot(gs[2])
text_ax.axis('off')

# Thêm vùng điều khiển
radio_ax = fig.add_axes([0.89, 0.7, 0.1, 0.12])
radio = RadioButtons(radio_ax, labels=['Distance', 'n_clusters'], active=0)

# Vùng nhập textbox + button
widget_box_ax = fig.add_axes([0.89, 0.6, 0.1, 0.05])
widget_btn_ax = fig.add_axes([0.89, 0.52, 0.1, 0.05])

text_box = TextBox(widget_box_ax, 'Value:', initial="50")
btn = Button(widget_btn_ax, 'OK')

# Hàm xử lý khi nhấn nút OK
def update_clusters_from_input(val):
    global scatter
    try:
        mode = radio.value_selected  # Lấy lựa chọn từ RadioButtons
        input_value = float(val)

        if input_value <= 0:
            return

        if mode == 'Distance':
            cut_line.set_ydata([input_value])  # Cập nhật đường cắt
            clusters = cut_dendrogram(linked, distance_threshold=input_value)
        elif mode == 'n_clusters':
            # Tìm khoảng cách cao nhất để còn đúng n_clusters cụm
            dists = np.sort(linked[:, 2])
            best_cut = None

            for d in dists:
                clusters_tmp = fcluster(linked, d, criterion='distance')
                if len(np.unique(clusters_tmp)) == int(input_value):
                    best_cut = d
                elif len(np.unique(clusters_tmp)) < int(input_value):
                    break  # vượt qua rồi, không cần xét thêm

            if best_cut is not None:
                cut_line.set_ydata([best_cut])
            clusters = cut_dendrogram(linked, n_clusters=int(input_value))
        else:
            return

        n_clusters = len(np.unique(clusters))
        cluster_colors = [colors[i % len(colors)] for i in range(n_clusters)]
        color_map = [cluster_colors[cluster - 1] for cluster in clusters]

        scatter.remove()
        scatter = ax_scatter.scatter(
            df.iloc[:, 0], df.iloc[:, 1],
            c=color_map, edgecolor='none'
        )

        fig.canvas.draw_idle()

    except ValueError:
        print("Please enter a valid number.")

btn.on_clicked(lambda event: update_clusters_from_input(text_box.text))
plt.show()


#---------------------------- SỬ DỤNG T-SNE --------------------------------#
# Sau khi cắt cây và có "clusters"
y_pred = cut_dendrogram(linked, n_clusters=2)

# Tính số lượng mỗi nhóm
unique_labels, counts = np.unique(y_pred, return_counts=True)
print("Number of elements in each cluster: ")
for label, count in zip(unique_labels, counts):
    print(f"  Cluster {label}: {count}")

# Sử dụng t-SNE để giảm chiều dữ liệu xuống 2D
tsne = TSNE(n_components=2, random_state=42)

# Áp dụng t-SNE lên dữ liệu (hoặc bạn có thể sử dụng dữ liệu sau khi phân cụm)
X_tsne = tsne.fit_transform(df.iloc[:, :2])

# Vẽ biểu đồ scatter với t-SNE
plt.figure(figsize=(8, 6))
scatter_tsne = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_pred, cmap='tab10', edgecolor='none')

# Thêm tiêu đề và nhãn cho trục
plt.title('t-SNE Visualization for Hierarchical Clustering')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')

# Lưu biểu đồ
output_path = 'data/output/tsne_hierarchical_clustering.png'
plt.savefig(output_path)
print(f"Saved t-SNE graph at: {output_path}")

plt.show()