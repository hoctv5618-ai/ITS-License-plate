# 🚗 Hệ thống Nhận diện Biển số Xe Việt Nam trong Giao thông Thông minh (ITS)

Dự án xây dựng đường ống (pipeline) đầu-cuối nhận diện, theo dõi và quản lý biển số xe Việt Nam từ **Ảnh tĩnh** và **Video giao thông**:

> **Ảnh / Video đầu vào** ➔ **Phát hiện biển số (`YOLOv8n`)** ➔ **Cắt vùng biển (lề 5%) & Lọc khung hình tốt nhất (`Best-Shot`)** ➔ **Tiền xử lý (`Xám` + `Xoay phẳng Hough` + `Cắt viền` + `Resize giữ tỉ lệ` + `CLAHE`)** ➔ **Đọc ký tự (`EasyOCR` + `Gom dòng` + `Đọc 2 nửa`)** ➔ **Kiểm tra & Sửa lỗi định dạng (`validate.py`)** ➔ **Bỏ phiếu theo thời gian (`Temporal Voting`)** ➔ **Giao diện Web `Streamlit` & Lưu trữ `SQLite`**.

---

## MỤC LỤC

1. [Yêu cầu & Cài đặt môi trường](#1-yêu-cầu--cài-đặt-môi-trường)
2. [Khởi chạy nhanh ứng dụng](#2-khởi-chạy-nhanh-ứng-dụng)
3. [Tác dụng chi tiết của từng thư mục và tệp tin](#3-tác-dụng-chi-tiết-của-từng-thư-mục-và-tệp-tin)
4. [Hướng dẫn sử dụng giao diện ứng dụng (Streamlit)](#4-hướng-dẫn-sử-dụng-giao-diện-ứng-dụng-streamlit)
5. [Hướng dẫn thực thi các quy trình kỹ thuật bằng dòng lệnh (CLI)](#5-hướng-dẫn-thực-thi-các-quy-trình-kỹ-thuật-bằng-dòng-lệnh-cli)
6. [Bảng công tắc cấu hình (`config.py`) & Phạm vi hỗ trợ](#6-bảng-công-tắc-cấu-hình-configpy--phạm-vi-hỗ-trợ)

---

## 1. Yêu cầu & Cài đặt môi trường

- **Hệ điều hành:** Windows 10/11 (hỗ trợ đầy đủ đường dẫn tiếng Việt có dấu).
- **Python:** 3.10 trở lên (khuyến nghị có GPU NVIDIA hỗ trợ CUDA để tăng tốc YOLO và EasyOCR).

```powershell
# Bước 1: Cài đặt PyTorch hỗ trợ CUDA (ví dụ CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Bước 2: Cài đặt các thư viện cần thiết của dự án
pip install ultralytics opencv-python numpy pandas matplotlib imagehash Pillow easyocr streamlit
```

---

## 2. Khởi chạy nhanh ứng dụng

Mở **PowerShell** hoặc **Terminal**, di chuyển vào thư mục `D:\ITS\plate_app` và chạy:

```powershell
cd D:\ITS\plate_app
python -m streamlit run app.py
```

*(Hoặc chạy trực tiếp `python app.py` — ứng dụng sẽ tự động gọi `streamlit run` và mở trình duyệt tại địa chỉ `http://localhost:8501`).*

---

## 3. Tác dụng chi tiết của từng thư mục và tệp tin

### 3.1. Thư mục gốc `D:\ITS\`

| Thư mục / Tệp tin | Tác dụng chi tiết |
|---|---|
| `ITS/` | **Thư mục dữ liệu gốc (Raw Data):** Chứa các bộ dữ liệu tải về trước khi chuẩn hóa, gồm:<br>• `kaggle_vnlp/`: Bộ dữ liệu biển số xe Việt Nam từ Kaggle (498 ảnh, chủ yếu độ phân giải Full HD `1920×1080`, nhiều biển xe máy 2 dòng).<br>• `roboflow_cuongta/`: Bộ dữ liệu biển số xuất từ Roboflow định dạng YOLOv8 (8.254 ảnh, chủ yếu `472×303` từ camera trạm kiểm soát).<br>• `unused_vehicles/`: Bộ dữ liệu phương tiện không dùng trong bài toán phát hiện biển số (tách riêng để tránh lẫn nhãn). |
| `plate_app/` | **Thư mục mã nguồn chính của dự án:** Chứa toàn bộ code xử lý, mô hình đã huấn luyện, cơ sở dữ liệu, công cụ gán nhãn, báo cáo và tài liệu. |
| `yolov8n.pt` | Trọng số khởi tạo gốc (pretrained trên tập COCO) của kiến trúc YOLOv8 Nano, dùng làm điểm bắt đầu khi huấn luyện. |
| `weights/` | Thư mục lưu trữ trọng số dự phòng ngoài luồng chính (`yolo26n.pt` — không sử dụng trong pipeline hiện tại). |
| `runs/` | Thư mục mặc định do thư viện Ultralytics sinh ra khi chạy lệnh kiểm thử ngoài thư mục `plate_app`. |
| `README.md` | Tệp giới thiệu ngắn ở thư mục gốc, dẫn hướng vào `plate_app/README.md`. |

---

### 3.2. Các thư mục con bên trong `D:\ITS\plate_app\`

```text
D:\ITS\plate_app\
├── data/                       # Trung tâm lưu trữ dữ liệu đã xử lý, bộ kiểm thử OCR, CSDL SQLite và ảnh kết quả
│   ├── processed/
│   │   ├── all/                # 8.752 ảnh (images/) và 8.752 nhãn (labels/) đã kiểm tra hợp lệ và gộp từ 2 nguồn
│   │   └── split/              # Dữ liệu chia cụm chống rò rỉ (train: 7.001 | val: 876 | test: 875 ảnh)
│   ├── ocr_set/                # 150 ảnh biển số cắt từ tập test (lấy mẫu phân tầng 12 ô) để chấm điểm OCR
│   ├── results/                # Lưu ảnh gốc, ảnh biển cắt, khung hình Best-Shot và video .webm xuất ra từ giao diện
│   ├── data.yaml               # File cấu hình đường dẫn train/val/test và nhãn lớp ('license_plate') cho YOLOv8
│   ├── ocr_labels.csv          # Bảng nhãn của 150 ảnh OCR (chia subset dev/final và đánh dấu excluded=1)
│   ├── plates_history.db       # Cơ sở dữ liệu SQLite lưu toàn bộ lịch sử nhận diện biển số (bảng detections)
│   ├── plates_history_backup.db# Bản sao lưu tự động của CSDL trước khi xóa dữ liệu thử nghiệm
│   ├── test_kaggle.txt         # Danh sách 50 đường dẫn ảnh test thuộc nguồn Kaggle (để đánh giá tách nguồn)
│   └── test_roboflow.txt       # Danh sách 825 đường dẫn ảnh test thuộc nguồn Roboflow
├── models/                     # Lưu trữ trọng số mô hình phát hiện biển số đã huấn luyện
│   ├── best_n640.pt            # Trọng số chính thức (80 epoch, dừng sớm ở epoch 69, tốt nhất ở epoch 54 - mAP50 = 0.9938)
│   └── best_n640_quick.pt      # Trọng số chạy thử nhanh (3 epoch - dùng để đối chiếu tốc độ hội tụ)
├── reports/                    # Chứa toàn bộ báo cáo thực nghiệm, bảng CSV thống kê và ảnh minh họa
│   ├── detector_failures/      # Lưu ảnh mà YOLO bỏ sót (FN) hoặc khoanh nhầm (FP) trên tập test (best_n640 chỉ còn 22 ảnh)
│   ├── duplicate_samples/      # 10 ảnh ghép minh họa các cặp ảnh trùng/gần trùng tìm được bằng MD5 & pHash
│   ├── source_stats.csv        # Bảng thống kê chất lượng dữ liệu đầu vào của 2 nguồn Kaggle & Roboflow
│   ├── dropped.csv             # Danh sách ảnh bị loại do lỗi định dạng/hỏng file (0 ảnh bị loại)
│   ├── duplicates.csv          # Danh sách 4.550 cặp ảnh trùng tuyệt đối (MD5) hoặc gần trùng (pHash <= 6)
│   ├── big_group_sample.jpg    # Ảnh minh họa cụm ảnh gần trùng lớn nhất (kiểm tra tính chất bắc cầu Union-Find)
│   ├── ratio_hist.png          # Biểu đồ phân bố tỉ lệ Rộng/Cao (w/h) giúp chọn ngưỡng phân tách biển 1 dòng / 2 dòng (2.1)
│   ├── io_test_sample.jpg      # Ảnh kiểm chứng đọc/ghi thành công trên đường dẫn tiếng Việt
│   ├── preprocess_demo.jpg     # Ảnh ghép minh họa trực quan 4 bước tiền xử lý trên 3 biển vuông và 3 biển dài
│   ├── train_results.md        # Báo cáo tự động ghi lại cấu hình, độ chính xác Val/Test và tốc độ của từng lần train
│   ├── compare_detectors.csv   # Bảng đối chiếu chi tiết các mô hình YOLO trên tập test (tách theo chiều cao biển <30, 30-60, >60px)
│   ├── ocr_raw_cache.csv       # Bộ nhớ đệm lưu kết quả chữ OCR thô (để không phải chạy lại EasyOCR khi thử nghiệm luật)
│   ├── ocr_comparison.csv      # Bảng so sánh 3 chế độ tiền xử lý (raw, clahe, threshold) x 2 chế độ validate
│   ├── ocr_ablation.csv        # Bảng khảo sát cắt bỏ từng nhóm luật hậu xử lý (Ablation Study) trên tập dev
│   └── ocr_ablation_preprocess.csv # Bảng khảo sát ảnh hưởng của từng công tắc tiền xử lý (xoay Hough, cắt mép, đọc 2 nửa)
├── runs/                       # Chứa nhật ký và biểu đồ quá trình huấn luyện YOLOv8
│   ├── detect/n640/            # Biểu đồ loss, PR_curve, confusion_matrix và weights của lần huấn luyện chính thức
│   ├── detect/n640_quick/      # Biểu đồ và kết quả của lần chạy thử 3 epoch
│   └── train_n640.log          # File log chi tiết toàn bộ quá trình huấn luyện
├── tools/                      # Bộ công cụ độc lập phục vụ phân tích dữ liệu, gán nhãn và quản trị
│   ├── plot_ratio.py           # Vẽ biểu đồ histogram tỉ lệ w/h của khung biển số trên tập train
│   ├── image_sizes.py          # Phân tích kích thước ảnh gốc và phân bố chiều cao biển số (pixel)
│   ├── make_ocr_labelset.py    # Trích xuất phân tầng 150 ảnh biển số từ tập test và chia sẵn subset dev/final
│   ├── label_tool.py           # Ứng dụng web Streamlit hỗ trợ gõ nhãn tay nhanh cho bộ kiểm thử OCR
│   └── clear_history.py        # Script sao lưu CSDL sang plates_history_backup.db rồi làm sạch dữ liệu thử nghiệm
└── docs/                       # Tài liệu học thuật và báo cáo dự án
    ├── BAO_CAO_DU_AN.md        # Báo cáo tổng hợp toàn diện về dự án, số liệu thực nghiệm và hạn chế
    ├── GIAI_THICH.md           # Giải thích chi tiết từng thuật toán kèm ví dụ số nhỏ (dùng cho báo cáo & bảo vệ đồ án)
    └── NOTES_CHO_GIAI_DOAN_3.md# Ghi chú thiết kế chuyển tiếp từ giai đoạn phát hiện sang giai đoạn OCR
```

---

### 3.3. Tác dụng của từng tệp mã nguồn Python (`.py`) trong `plate_app/`

| Phân nhóm | Tệp mã nguồn | Tác dụng chi tiết |
|---|---|---|
| **Cấu hình & Đọc/Ghi** | `config.py` | Khai báo tập trung mọi đường dẫn tương đối (tự động nhận diện ổ đĩa), ngưỡng độ tự tin YOLO/OCR, tham số CLAHE, mẫu biểu thức chính quy (`PLATE_PATTERNS`) và các công tắc bật/tắt luật (`RULE_*`). |
| | `image_io.py` | Cung cấp hàm `read_image()` và `save_image()` dùng `np.fromfile` + `cv2.imdecode` / `cv2.imencode`, khắc phục triệt để lỗi OpenCV không đọc được đường dẫn có dấu tiếng Việt trên Windows. |
| **Giai đoạn 0: Dữ liệu** | `prepare_data.py` | Quét thư mục dữ liệu thô, kiểm tra tính toàn vẹn của từng cặp ảnh - nhãn YOLO, chuẩn hóa tên file theo nguồn (`kaggle_`, `roboflow_`), gộp vào `data/processed/all/` và xuất thống kê `source_stats.csv`. |
| | `find_duplicates.py` | Phát hiện ảnh trùng tuyệt đối bằng mã băm `MD5` và ảnh gần trùng bằng mã băm cảm nhận `pHash` (khoảng cách Hamming $\le 6$ tính bằng phép `XOR`), xuất ra `reports/duplicates.csv`. |
| | `split_dataset.py` | Gom các cặp ảnh trùng thành từng cụm liên thông bằng cấu trúc `Union-Find`, chia tập theo cụm với tỉ lệ `80% train / 10% val / 10% test` riêng cho từng nguồn để chống rò rỉ dữ liệu (Data Leakage), đồng thời sinh file `data/data.yaml`. |
| **Giai đoạn 1: Tiền xử lý** | `preprocess.py` | Chuyển ảnh biển cắt sang ảnh xám (`to_grayscale`), xoay phẳng biển bị nghiêng bằng biến đổi Hough (`deskew_plate`), cắt bỏ viền kim loại 2 bên với biển dài, phóng to giữ nguyên tỉ lệ nếu chiều cao $< 100\text{px}$ (`resize_plate`), cân bằng sáng cục bộ `CLAHE` (`apply_clahe`) và tùy chọn tách nhị phân `Otsu` (`apply_threshold`). |
| **Giai đoạn 2: Phát hiện** | `train.py` | Huấn luyện mô hình `YOLOv8n` với cấu hình cố định (`seed=42`), tự động chấm điểm trên tập `val`, tập `test` chung, tách riêng kết quả trên `test_kaggle` / `test_roboflow` và nối báo cáo vào `reports/train_results.md`. |
| | `detect.py` | Tải trọng số `best_n640.pt`, dự đoán tọa độ khung bao `[x1, y1, x2, y2, conf]`, cắt vùng ảnh biển số kèm lề mở rộng 5% (`crop_plate`) và vẽ khung kết quả lên ảnh (`draw_detections`). |
| | `evaluate_utils.py` | Cài đặt hàm tính độ chồng lấp `compute_iou()`, thuật toán ghép cặp tham lam 1-1 (`match_detections`) phân loại `TP / FP / FN` và hàm vẽ khung báo lỗi. |
| | `evaluate.py` | Đánh giá chi tiết các file trọng số trong `models/` trên 875 ảnh test, thống kê `Recall` theo 3 mức chiều cao biển (`<30px`, `30-60px`, `>60px`), xuất bảng `compare_detectors.csv` và lưu toàn bộ ảnh lỗi vào `reports/detector_failures/`. |
| **Giai đoạn 3: Đọc & Kiểm tra biển** | `ocr.py` | Khởi tạo `EasyOCR` 1 lần duy nhất, lọc ký tự theo `OCR_ALLOWLIST`, gom các hộp chữ rời rạc thành từng dòng theo tọa độ dọc & độ chồng lấp ngang (`group_into_lines`), và tách đôi nửa trên/nửa dưới đọc lại với biển vuông 2 dòng (`read_two_halves`). |
| | `validate.py` | Làm sạch chuỗi thô (`normalize`), sửa lỗi ký tự nhầm giữa chữ và số (`0↔O, 1↔I, 2↔Z, 5↔S, 6↔G, 8↔B...`) theo mặt nạ vị trí từng loại biển (`apply_mask_and_format`), kiểm tra mã tỉnh (`11..99`) và trả về loại biển + trạng thái hợp lệ. |
| | `evaluate_ocr.py` | Đo độ chính xác khớp toàn bộ biển (`Exact Match %`), tỉ lệ lỗi ký tự (`CER %` qua khoảng cách chỉnh sửa `Levenshtein`), lưu cache chữ OCR thô (`ocr_raw_cache.csv`) và chạy khảo sát cắt bỏ luật (`ocr_ablation.csv`, `ocr_ablation_preprocess.csv`). |
| **Giai đoạn 4: Luồng & CSDL** | `database.py` | Quản lý cơ sở dữ liệu `SQLite` (`plates_history.db`) với truy vấn tham số hóa `?` chống SQL Injection: thêm bản ghi, lọc lịch sử đa điều kiện, tính thống kê tổng quan, xóa từng dòng và sao lưu trước khi xóa toàn bộ. |
| | `video_tracker.py` | Bộ theo dõi đa đối tượng trên video: ước lượng vận tốc di chuyển `(vx, vy)`, ghép cặp `Track ID` qua `IoU` và khoảng cách tâm, lọc khung hình rõ nét nhất (`check_best_shot` qua phương sai Laplacian), khóa OCR khi đủ 5 lần đọc hợp lệ, bỏ phiếu có trọng số và vẽ thanh trạng thái HUD lên video. |
| | `pipeline.py` | Kết nối toàn bộ các mô-đun thành 2 hàm điều phối chính: `process_image()` cho ảnh tĩnh và `process_video()` cho luồng video (kèm gộp các cụm trùng biển số khi camera lia qua lại và xuất video `.webm`). |
| **Giai đoạn 5: Giao diện Web** | `app.py` | Điểm vào chính của giao diện Streamlit: nạp mô hình vào bộ nhớ đệm (`@st.cache_resource`), tạo thanh cấu hình bên trái và điều phối 3 Tab chức năng. |
| | `ui_image.py` | Giao diện **Tab Ảnh tĩnh**: hỗ trợ tải nhiều ảnh cùng lúc, hiển thị ảnh đóng khung, ảnh cắt & ảnh sau tiền xử lý, cho phép sửa tay biển số bị nhầm để lưu vào CSDL và tải bảng tổng hợp CSV. |
| | `ui_video.py` | Giao diện **Tab Video**: tải video giao thông, hiển thị thanh tiến trình xử lý, trình phát video kết quả gắn `ID #`, hiển thị khung hình `Best-Shot` của từng xe và bảng tổng hợp. |
| | `ui_history.py` | Giao diện **Tab Lịch sử**: hiển thị 4 thẻ thống kê, bộ lọc tìm kiếm theo chuỗi biển/ngày/nguồn/trạng thái, xem lại ảnh gốc & ảnh cắt của từng lượt, xóa bản ghi lẻ hoặc sao lưu & xóa dữ liệu thử nghiệm. |

---

## 4. Hướng dẫn sử dụng giao diện ứng dụng (Streamlit)

Sau khi chạy lệnh `python -m streamlit run app.py`, giao diện web mở ra gồm **Thanh cấu hình bên trái (Sidebar)** và **3 Tab chức năng chính**:

### 4.1. Thanh cấu hình bên trái (Sidebar)
- **Ngưỡng độ tự tin YOLO (`DETECT_CONF`):** Thanh trượt từ `0.10` đến `0.90` (mặc định `0.40`).
  - *Mẹo:* Nếu ảnh có nhiều xe ở xa hoặc góc nghiêng (như ảnh 2 siêu xe), giữ ở mức `0.40` để bắt đủ cả các biển số có độ tự tin vừa phải. Nếu muốn giảm báo nhầm ở cảnh nhiều biển quảng cáo, có thể kéo lên `0.50 - 0.60`.
- **Bật tiền xử lý ảnh biển (Xám + Resize + CLAHE):** Mặc định bật (`✅`) để tăng độ nét và cân bằng sáng trước khi đọc chữ.
- **Tự động lưu kết quả vào SQLite:** Mặc định bật (`✅`). Mọi biển số phát hiện từ Ảnh hoặc Video sẽ tự động ghi vào `data/plates_history.db` và lưu ảnh ra `data/results/`.

---

### 4.2. Tab 1: 🖼️ Ảnh (Nhận diện trên một hoặc nhiều ảnh tĩnh)
1. Bấm **Browse files** (hoặc kéo thả) một hay nhiều file ảnh (`.jpg`, `.jpeg`, `.png`, `.webp`).
2. Với mỗi bức ảnh, màn hình chia làm 2 cột trực quan:
   - **Cột trái:** Ảnh gốc đã được vẽ khung bao màu xanh lá kèm chuỗi biển số và độ tự tin YOLO.
   - **Cột phải:** Hiển thị từng biển số phát hiện được trong ảnh, gồm:
     - Ảnh biển cắt (có nới lề 5%) và ảnh sau bước tiền xử lý (xoay phẳng + CLAHE).
     - Kết quả biển số đã chuẩn hóa, nhãn trạng thái (**✅ Hợp lệ** hoặc **⚠️ Không hợp lệ**), loại biển (`oto_5so`, `oto_4so`, `xemay_5so`, `xemay_4so`), mã tỉnh và thời gian xử lý chi tiết (`Detect ms`, `Prep ms`, `OCR ms`).
     - **Chức năng sửa tay:** Nếu ảnh quá mờ làm OCR đọc nhầm 1 ký tự, bạn có thể gõ lại biển đúng vào ô *"Sửa tay biển số"* rồi bấm **Lưu bản sửa tay vào CSDL** (bản ghi sẽ được đánh dấu `is_corrected = 1`).
3. Cuối trang là **Bảng tổng hợp kết quả lô ảnh** kèm nút **⬇️ Tải bảng kết quả CSV**.

---

### 4.3. Tab 2: 🎥 Video (Theo dõi ID #, Lọc Best-Shot & Bỏ phiếu đa khung hình)
1. Tải lên một file video giao thông (`.mp4`, `.avi`, `.mov`).
2. Chọn thông số **Xử lý mỗi N khung hình (`FRAME_SKIP`)**: mặc định là `3` (cứ 3 frame xử lý 1 frame để tăng tốc độ gấp 3 lần mà vẫn theo dõi mượt).
3. Bấm nút **▶️ Bắt đầu phân tích Video** và theo dõi thanh tiến trình.
4. Khi hoàn tất, màn hình hiển thị:
   - **Trình phát Video kết quả (`.webm`)** có vẽ thanh trạng thái HUD ở trên cùng và khung bao gắn `ID #<số> | <Biển số>` di chuyển theo từng xe, kèm nút tải video về máy.
   - **Thống kê tiết kiệm tài nguyên:** Tổng số xe/biển duy nhất, tổng số frame phát hiện so với số lần thực sự phải gọi mạng EasyOCR (nhờ cơ chế *Smart Best-Shot Trigger* và khóa ID sau 5 lần đọc hợp lệ).
   - **Danh sách thẻ Best-Shot của từng xe (`ID #`):** Hiển thị khung hình toàn cảnh rõ nét nhất (`best_frame`) có đóng khung đúng vị trí biển số tại thời điểm đó (`best_bbox`), ảnh biển cắt tương ứng, điểm độ nét Laplacian và kết quả bầu chọn cuối cùng.
   - **Bảng tổng hợp các phương tiện duy nhất** ở cuối trang.

---

### 4.4. Tab 3: 🗂️ Lịch sử (Tra cứu, Thống kê & Quản trị CSDL)
1. **4 Thẻ thống kê trên cùng:** Tổng số lượt nhận diện, tỉ lệ biển hợp lệ (%), Top 3 mã tỉnh xuất hiện nhiều nhất và thời gian xử lý trung bình (ms).
2. **Nút dọn dẹp dữ liệu thử nghiệm:** Bấm **🧹 Sao lưu & Xóa dữ liệu thử trong CSDL (lưu ra plates_history_backup.db)** để tự động tạo bản sao lưu an toàn trước khi làm trống bảng lịch sử.
3. **Bộ lọc tra cứu 4 tiêu chí:**
   - Tìm theo chuỗi ký tự biển số (ví dụ gõ `30A` hoặc `59-X1`).
   - Lọc theo ngày (`YYYY-MM-DD`).
   - Lọc theo nguồn (`Tất cả`, `Ảnh tĩnh`, `Video`).
   - Lọc theo trạng thái (`Tất cả`, `Hợp lệ`, `Không hợp lệ`).
4. **Xem chi tiết & Xóa lẻ:** Chọn một bản ghi bất kỳ theo `ID #` ở mục dưới bảng để đối chiếu lại **Ảnh gốc** và **Ảnh biển cắt** đã lưu trên ổ đĩa, hoặc bấm **🗑️ Xóa bản ghi ID #...** để xóa riêng dòng đó.
5. Bấm **⬇️ Xuất lịch sử ra CSV** để tải các bản ghi đang lọc về mở bằng Excel (hỗ trợ chuẩn tiếng Việt `UTF-8-SIG`).

---

## 5. Hướng dẫn thực thi các quy trình kỹ thuật bằng dòng lệnh (CLI)

Tất cả các lệnh dưới đây được thực thi khi đứng tại thư mục `D:\ITS\plate_app`:

```powershell
cd D:\ITS\plate_app
```

### 5.1. Kiểm thử nhanh toàn bộ các mô-đun cốt lõi (chạy bằng CPU)
Mỗi mô-đun đều có khối tự kiểm tra (`if __name__ == "__main__":`) để xác nhận hoạt động độc lập:

```powershell
python config.py          # Kiểm tra đường dẫn và khởi tạo thư mục
python validate.py        # Chạy 31 ca kiểm thử định dạng biển số (22 ca gốc + 9 ca từ clip -> 31/31 PASS)
python database.py        # Kiểm thử thêm, lọc, thống kê, xóa và sao lưu CSDL tạm (-> PASS)
python evaluate_utils.py  # Kiểm thử 11 ca tính IoU và ghép cặp tham lam (-> 11/11 PASS)
python ocr.py             # Kiểm thử thuật toán gom dòng chữ bị nghiêng
python pipeline.py        # Kiểm thử bộ theo dõi ID #, bỏ phiếu và đồng bộ tọa độ Best-Shot
```

### 5.2. Quy trình chuẩn bị dữ liệu, lọc trùng và chia tập (Giai đoạn 0)
*(Chỉ cần chạy lại khi bạn thêm dữ liệu thô mới vào thư mục `D:\ITS\ITS`):*

```powershell
python prepare_data.py     # Kiểm tra ảnh-nhãn, gộp về data/processed/all/ và xuất reports/source_stats.csv
python find_duplicates.py  # Quét MD5 & pHash tìm ảnh trùng, xuất reports/duplicates.csv
python split_dataset.py    # Gom cụm Union-Find, chia 80/10/10 chống rò rỉ và sinh data/data.yaml
python tools/plot_ratio.py # Vẽ biểu đồ phân bố tỉ lệ rộng/cao tại reports/ratio_hist.png
python tools/image_sizes.py# In thống kê phân bố độ phân giải ảnh gốc và chiều cao biển số
```

### 5.3. Quy trình huấn luyện và đánh giá mô hình phát hiện YOLOv8 (Giai đoạn 2)

```powershell
# 1. Huấn luyện mô hình (chỉnh QUICK_TEST = True/False ở đầu file train.py)
python train.py

# 2. Đánh giá toàn diện các trọng số trong models/ trên 875 ảnh test
python evaluate.py
```
- Sau khi chạy, xem báo cáo tại `reports/train_results.md`, bảng so sánh tại `reports/compare_detectors.csv` và các ảnh bị bỏ sót/báo nhầm tại `reports/detector_failures/best_n640/`.

### 5.4. Quy trình gán nhãn và đánh giá nhận dạng ký tự OCR (Giai đoạn 3)

```powershell
# Bước 1: Mở công cụ gán nhãn trên trình duyệt (tự động bỏ qua 7 ảnh excluded=1)
python -m streamlit run tools/label_tool.py

# Bước 2: Đánh giá trên tập dev (lưu cache chữ thô vào reports/ocr_raw_cache.csv,
#         xuất reports/ocr_comparison.csv và bảng khảo sát luật reports/ocr_ablation.csv)
python evaluate_ocr.py

# Bước 3: Khảo sát ảnh hưởng của từng công tắc tiền xử lý trên tập dev
#         (chạy lại EasyOCR cho từng cấu hình, xuất reports/ocr_ablation_preprocess.csv)
python evaluate_ocr.py --ablate-preprocess

# Bước 4: Đánh giá trên tập final (CHỈ CHẠY ĐÚNG 1 LẦN DUY NHẤT khi chốt số liệu báo cáo cuối)
python evaluate_ocr.py --final
```

### 5.5. Dọn dẹp dữ liệu thử nghiệm trong CSDL bằng dòng lệnh

```powershell
python tools/clear_history.py
```
*(Tự động sao lưu `data/plates_history.db` sang `data/plates_history_backup.db` rồi xóa sạch các bản ghi thử nghiệm).*

---

## 6. Bảng công tắc cấu hình (`config.py`) & Phạm vi hỗ trợ

Để phục vụ nghiên cứu thực nghiệm và khảo sát cắt bỏ (Ablation Study), mọi quy tắc tiền xử lý và hậu xử lý bổ sung đều có công tắc bật/tắt độc lập trong `config.py`:

| Hằng số trong `config.py` | Giá trị mặc định | Ý nghĩa & Tác dụng |
|---|---|---|
| `MAX_FIXED_CHARS` | `3` | Số ký tự tối đa cho phép sửa nhầm chữ ↔ số trên một biển số trong `validate.py`. |
| `RULE_STRAIGHTEN` | `True` | Tự động xoay phẳng ảnh biển số bị nghiêng theo góc trung vị đường thẳng Hough (`preprocess.py`). |
| `RULE_TRIM_EDGE` | `True` | Cắt bỏ 4.5% mép trái/phải và 4% mép trên/dưới đối với biển dài 1 dòng để loại bỏ viền kim loại (`preprocess.py`). |
| `RULE_TWO_HALVES` | `True` | Tách đôi nửa trên (`0..55%`) và nửa dưới (`45..100%`) đọc lại khi biển vuông 2 dòng chưa hợp lệ ở lần đọc đầu (`ocr.py`). |
| `RULE_STRIP_WATERMARK` | `True` | Loại bỏ chuỗi watermark `TIKTOK`/`TKTOK` đè lên góc biển số trong video (`validate.py`). |
| `RULE_SWAP_INVERTED_LINES` | `True` | Tự động đảo lại thứ tự khi dòng số thứ tự `XXX.XX` bị đọc trước dòng mã tỉnh + seri trên biển 2 dòng nghiêng (`validate.py`). |
| `RULE_TRIM_TRAILING_BORDER`| `True` | Cắt 1 ký tự nét dọc thừa thuộc nhóm `[17IJT]` bám sau cụm `.XX` ở cuối biển do viền khung/ốc vít (`validate.py`). |
| `RULE_EXTRA_DIGIT_MAP` | `True` | Bật bảng ánh xạ bổ sung sang chữ số `EXTRA_TO_DIGIT = {'Q':'0', 'U':'0', 'J':'3', 'L':'4', 'T':'7'}` bên cạnh bảng gốc `BASE_TO_DIGIT` (`validate.py`). |

### Lưu ý về phạm vi hỗ trợ định dạng biển số
- Các mẫu định dạng trong `config.PLATE_PATTERNS` và `validate.py` được xây dựng **dựa trên định dạng biển phổ biến quan sát được, chưa đối chiếu văn bản quy định**:
  - Ô tô 5 số (`30A-123.45`), Ô tô 4 số cũ (`29A-1234`).
  - Xe máy 2 dòng 5 số (`59-X1 123.45`, `43-AA 567.89`), Xe máy 2 dòng 4 số cũ (`29-L1 1234`).
- **Hạn chế:** Các loại biển 4 số có 2 chữ cái, biển quân đội, biển ngoại giao và biển công vụ (ví dụ `43-346-CV-03`) hiện **chưa được hỗ trợ** (hệ thống vẫn đọc và hiển thị chuỗi ký tự thô nhưng sẽ đánh dấu trạng thái là `Không hợp lệ`).
- Xem thêm phân tích thuật toán chi tiết tại [`../docs/GIAI_THICH.md`](../docs/GIAI_THICH.md) và báo cáo tổng hợp dự án tại [`../docs/BAO_CAO_DU_AN.md`](../docs/BAO_CAO_DU_AN.md).
- Xem bảng theo dõi tiến độ WBS từng tuần và phân công thành viên tại [`TIEN_DO_WBS.md`](TIEN_DO_WBS.md).
