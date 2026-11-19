
## Lần chạy: `n640_quick` (2026-10-30 16:58)
1. **Cấu hình:** Model gốc: `yolov8n.pt` | Epochs đặt/dừng: `3/3` | `imgsz=640`, `batch=16`, `seed=42`, `WORKERS=2`, `PATIENCE=15` | Thiết bị: `NVIDIA GeForce RTX 3050 6GB Laptop GPU` | Thời gian: `5.06` phút.
   - *Lưu ý: So sánh 640 và 960 chỉ có ý nghĩa khi cùng seed, cùng model, cùng epochs.*
2. **Dữ liệu:** Train `7001`, Val `876`, Test `875` | File: `D:/ITS/plate_app/data/data.yaml` | Kích thước gốc phổ biến nhất (train): `kaggle=1920x1080`, `roboflow=472x303`.
3. **Val tốt nhất (Epoch 3):** Precision `0.9846`, Recall `0.9768`, mAP50 `0.9936`, mAP50-95 `0.7056`.
4. **Test chung:** Precision `0.9810`, Recall `0.9628`, mAP50 `0.9800`, mAP50-95 `0.6855`.
5. **Test tách theo nguồn** *(Tập test kaggle chỉ khoảng 50 ảnh nên số liệu dao động mạnh)*:

| Nguồn | Số ảnh | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| kaggle | 50 | 0.9427 | 0.9189 | 0.9141 | 0.7241 |
| roboflow | 825 | 0.9843 | 0.9697 | 0.9911 | 0.6841 |

6. **Tốc độ suy luận trung bình:** `3.56` ms/ảnh.
7. **Trọng số & Biểu đồ:** `D:/ITS/plate_app/models/best_n640_quick.pt`
   - ![results](../runs/detect/n640_quick/results.png) ![confusion](../runs/detect/n640_quick/confusion_matrix.png) ![PR](../runs/detect/n640_quick/PR_curve.png)
8. **Nhận xét:** 
---

## Lần chạy: `n640` (2026-10-30 18:32)
1. **Cấu hình:** Model gốc: `yolov8n.pt` | Epochs đặt/dừng: `80/69` | `imgsz=640`, `batch=16`, `seed=42`, `WORKERS=2`, `PATIENCE=15` | Thiết bị: `NVIDIA GeForce RTX 3050 6GB Laptop GPU` | Thời gian: `83.18` phút.
   - *Lưu ý: So sánh 640 và 960 chỉ có ý nghĩa khi cùng seed, cùng model, cùng epochs.*
2. **Dữ liệu:** Train `7001`, Val `876`, Test `875` | File: `D:/ITS/plate_app/data/data.yaml` | Kích thước gốc phổ biến nhất (train): `kaggle=1920x1080`, `roboflow=472x303`.
3. **Val tốt nhất (Epoch 54):** Precision `0.9923`, Recall `0.9941`, mAP50 `0.9940`, mAP50-95 `0.7524`.
4. **Test chung:** Precision `0.9967`, Recall `0.9846`, mAP50 `0.9938`, mAP50-95 `0.7340`.
5. **Test tách theo nguồn** *(Tập test kaggle chỉ khoảng 50 ảnh nên số liệu dao động mạnh)*:

| Nguồn | Số ảnh | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| kaggle | 50 | 0.9836 | 0.9189 | 0.9488 | 0.8114 |
| roboflow | 825 | 0.9976 | 0.9905 | 0.9950 | 0.7255 |

6. **Tốc độ suy luận trung bình:** `3.39` ms/ảnh.
7. **Trọng số & Biểu đồ:** `D:/ITS/plate_app/models/best_n640.pt`
   - ![results](../runs/detect/n640/results.png) ![confusion](../runs/detect/n640/confusion_matrix.png) ![PR](../runs/detect/n640/PR_curve.png)
8. **Nhận xét:** 
---
