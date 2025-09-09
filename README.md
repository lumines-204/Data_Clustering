# Clustering Algorithms on Breast Cancer Dataset

## Giới thiệu
Dự án này triển khai và so sánh ba thuật toán phân cụm phổ biến trên tập dữ liệu **Breast Cancer**:
- **K-Means Clustering** + **K_Mean++**
- **Hierarchical Clustering**
- **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**

Mục tiêu chính là trực quan hóa và đánh giá sự khác biệt giữa các phương pháp phân cụm khi áp dụng lên dữ liệu thực tế.

---

## Cấu trúc thư mục
```
Data_Clustering/
│── dbscan_clustering.py          # Cài đặt và chạy DBSCAN
│── hierarchical_clustering.py    # Cài đặt và chạy Hierarchical Clustering
│── kmean_clustering.py           # Cài đặt và chạy K-Means
│
└── data/
    ├── input/
    │   ├── breast_cancer_dataset.csv   # Dữ liệu gốc
    │   ├── data_description.txt        # Mô tả dữ liệu
    │   └── data_source_link.txt        # Link nguồn dữ liệu
    │
    └── output/
        ├── clustering_scatter_origin.png
        ├── kmeans_2d_plot.png
        ├── tsne_dbscan_clustering.png
        └── tsne_hierarchical_clustering.png
```

---

## Cài đặt
### Yêu cầu
- Python >= 3.8
- Các thư viện cần thiết:
  ```bash
  pip install numpy pandas matplotlib scikit-learn seaborn
  ```

---

## Cách chạy
### 1. K-Means
```bash
python Data_Clustering/kmean_clustering.py
```

### 2. DBSCAN
```bash
python Data_Clustering/dbscan_clustering.py
```

### 3. Hierarchical Clustering
```bash
python Data_Clustering/hierarchical_clustering.py
```

Kết quả sẽ được lưu trong thư mục `Data_Clustering/data/output/`.

---

## Kết quả
Một số ví dụ trực quan hóa từ dự án:

- **K-Means 2D Scatter Plot**  
  ![KMeans](Data_Clustering/data/output/kmeans_2d_plot.png)

- **DBSCAN với TSNE**  
  ![DBSCAN](Data_Clustering/data/output/tsne_dbscan_clustering.png)

- **Hierarchical Clustering với TSNE**  
  ![Hierarchical](Data_Clustering/data/output/tsne_hierarchical_clustering.png)

---

## Dữ liệu
- Tập dữ liệu **Breast Cancer Dataset**.  
- Mô tả chi tiết có trong file [`data/input/data_description.txt`](Data_Clustering/data/input/data_description.txt).  
- Nguồn dữ liệu được trích dẫn trong [`data/input/data_source_link.txt`](Data_Clustering/data/input/data_source_link.txt).
