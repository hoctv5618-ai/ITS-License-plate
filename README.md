# 🚗 Dự án Nhận diện Biển số Xe Việt Nam trong Hệ thống Giao thông Thông minh (ITS)

![Tiến độ](https://img.shields.io/badge/Ti%E1%BA%BFn%20%C4%91%E1%BB%99-Tu%E1%BA%A7n%2012%2F12%20(23%2F11%2F2026)-blue)
![Python](https://img.shields.io/badge/Python-3.11-yellow)
![YOLOv8](https://img.shields.io/badge/YOLOv8n-mAP50%200.9938-brightgreen)
![OCR](https://img.shields.io/badge/EasyOCR-33.33%25%20final%20%7C%2058.82%25%20%3E60px-orange)

Mã nguồn chính, trọng số phát hiện biển số (`models/best_n640.pt`), dữ liệu đánh giá OCR và công cụ thực nghiệm của ứng dụng nằm tại thư mục [`plate_app/`](plate_app/README.md).

---

## 1. Cài đặt & Khởi chạy nhanh ứng dụng (Giao diện Web Streamlit)

```bash
# 1. Cài đặt thư viện cần thiết (khuyến nghị Python 3.11)
pip install -r requirements.txt

# 2. Di chuyển vào thư mục ứng dụng và chạy Streamlit
cd plate_app
python -m streamlit run app.py
```

---

## 2. Cấu trúc thư mục Repository

| Thư mục / Tệp tin | Tác dụng |
| :--- | :--- |
| [`plate_app/`](plate_app/README.md) | **Thư mục ứng dụng chính:** Chứa mã nguồn Python (`app.py`, `detect.py`, `preprocess.py`, `ocr.py`, `validate.py`, `video_tracker.py`, `pipeline.py`, `database.py`), trọng số [`models/best_n640.pt`](plate_app/models/best_n640.pt), nhãn OCR ([`data/ocr_labels.csv`](plate_app/data/ocr_labels.csv)), báo cáo thực nghiệm (`reports/`) và công cụ (`tools/`). |
| [`docs/`](docs/BAO_CAO_DU_AN.md) | **Tài liệu kỹ thuật & báo cáo:** Chứa báo cáo tổng hợp ([`BAO_CAO_DU_AN.md`](docs/BAO_CAO_DU_AN.md)), giải thích thuật toán ([`GIAI_THICH.md`](docs/GIAI_THICH.md)) và ghi chú kỹ thuật ([`NOTES_CHO_GIAI_DOAN_3.md`](docs/NOTES_CHO_GIAI_DOAN_3.md)). |
| [`TIEN_DO_WBS.md`](TIEN_DO_WBS.md) | **Bảng WBS & Kiểm tra tiến độ từng tuần (T8–T12):** Checklist chi tiết, minh chứng thực nghiệm và nhật ký nộp bài của 5 thành viên. |
| [`requirements.txt`](requirements.txt) | Danh sách thư viện Python và phiên bản đã kiểm thử trên máy nhóm. |
| [`.gitignore`](.gitignore) | Loại trừ ảnh thô nặng (`ITS/`), bộ nhớ đệm (`__pycache__/`), nhưng giữ lại [`best_n640.pt`](plate_app/models/best_n640.pt) và [`ocr_labels.csv`](plate_app/data/ocr_labels.csv) để chạy kiểm chứng lại được ngay. |

---

## 3. Tài liệu hướng dẫn & Báo cáo chi tiết

- 📘 **Hướng dẫn cài đặt, sử dụng giao diện và lệnh CLI:** [`plate_app/README.md`](plate_app/README.md)
- 📊 **Báo cáo tổng hợp dự án & quy trình xử lý:** [`docs/BAO_CAO_DU_AN.md`](docs/BAO_CAO_DU_AN.md)
- 🧠 **Giải thích chi tiết các thuật toán (kèm ví dụ số):** [`docs/GIAI_THICH.md`](docs/GIAI_THICH.md)
- 📅 **Cấu trúc WBS & Đánh giá tiến độ từng tuần (T8–T12):** [`TIEN_DO_WBS.md`](TIEN_DO_WBS.md)

---

## 4. Phân công 5 Thành viên & Danh mục Tệp nộp trên GitHub

| Thành viên | Vai trò & Phần tiểu luận phụ trách | Tệp mã nguồn / Minh chứng nộp trên GitHub | Hoàn thành |
| :--- | :--- | :--- | :---: |
| **TV1 — Lê Khánh Hưng** | Phần I (Giới thiệu); Quy trình xử lý & Demo code/app (Mục 3.3–3.5, Chương 4); Hạn chế & Hướng phát triển (Mục 5.5, Phần III) | [`README.md`](README.md), [`TIEN_DO_WBS.md`](TIEN_DO_WBS.md), [`requirements.txt`](requirements.txt), [`plate_app/app.py`](plate_app/app.py), [`config.py`](plate_app/config.py), [`pipeline.py`](plate_app/pipeline.py), [`video_tracker.py`](plate_app/video_tracker.py), [`database.py`](plate_app/database.py), [`ui_image.py`](plate_app/ui_image.py), [`ui_video.py`](plate_app/ui_video.py), [`ui_history.py`](plate_app/ui_history.py), [`best_n640.pt`](plate_app/models/best_n640.pt) | ✅ 100% |
| **TV2 — Trần Văn Học** (`052206005618`) | Chương 1 (Tổng quan ITS & Nhận diện biển số); Soạn 10 câu hỏi & Dẫn Minigame (Phần IV) | [`docs/BAO_CAO_DU_AN.md`](docs/BAO_CAO_DU_AN.md), [`plate_app/README.md`](plate_app/README.md), [`plate_app/reports/hinh_bao_cao/`](plate_app/reports/hinh_bao_cao) *(Hình 3.7, 3.8, 4.2–4.4)* | ✅ 100% |
| **TV3 — Lê Tăng Minh Quân (Nhóm trưởng)** | Chương 2 (Cơ sở lý thuyết AI/CV/OCR); Nguồn & Tiền xử lý dữ liệu (Mục 3.1–3.2) | [`docs/GIAI_THICH.md`](docs/GIAI_THICH.md), [`prepare_data.py`](plate_app/prepare_data.py), [`find_duplicates.py`](plate_app/find_duplicates.py), [`split_dataset.py`](plate_app/split_dataset.py), [`preprocess.py`](plate_app/preprocess.py), [`image_io.py`](plate_app/image_io.py), [`source_stats.csv`](plate_app/reports/source_stats.csv), [`duplicates.csv`](plate_app/reports/duplicates.csv), [`ratio_hist.png`](plate_app/reports/ratio_hist.png) | ✅ 100% |
| **TV4 — Ngô Thành Phát** (`079206024228`) | Chương 5 (Kiểm thử, đánh giá thực nghiệm YOLOv8n/OCR, đo tốc độ & Phân tích 22 ảnh lỗi) | [`train.py`](plate_app/train.py), [`detect.py`](plate_app/detect.py), [`ocr.py`](plate_app/ocr.py), [`validate.py`](plate_app/validate.py), [`evaluate.py`](plate_app/evaluate.py), [`evaluate_ocr.py`](plate_app/evaluate_ocr.py), [`ocr_labels.csv`](plate_app/data/ocr_labels.csv), [`compare_detectors.csv`](plate_app/reports/compare_detectors.csv), [`ocr_comparison.csv`](plate_app/reports/ocr_comparison.csv), [`detector_failures_analysis.csv`](plate_app/reports/detector_failures_analysis.csv), [`bench_video.py`](plate_app/tools/bench_video.py) | ✅ 100% |
| **TV5 — Nguyễn Hoàng Phúc** (`060206005778`) | Chương 6 (Ứng dụng thực tế ITS, Pháp lý & Thách thức); Soạn 10 câu hỏi & Hỗ trợ Minigame (Phần IV) | [`docs/NOTES_CHO_GIAI_DOAN_3.md`](docs/NOTES_CHO_GIAI_DOAN_3.md), [`plate_app/data/data.yaml`](plate_app/data/data.yaml), [`tools/clear_history.py`](plate_app/tools/clear_history.py), nội dung Chương 6 & câu hỏi thảo luận | ✅ 100% |

---

## 5. Tóm tắt Tiến độ Thực hiện Theo Tuần (Cập nhật 23/11/2026)

> Chi tiết checklist từng tuần, từng thành viên và nhật ký nộp bài xem tại [`TIEN_DO_WBS.md`](TIEN_DO_WBS.md).

| Tuần & Mốc thời gian | Nội dung trọng tâm theo WBS | Thành viên nộp | Sản phẩm minh chứng | Trạng thái |
| :--- | :--- | :---: | :--- | :---: |
| **T8 (30/10/2026)** | Chốt đề tài, WBS, làm sạch & chia tập 8.752 ảnh | TV1, TV3 | [`source_stats.csv`](plate_app/reports/source_stats.csv), [`duplicates.csv`](plate_app/reports/duplicates.csv) | ✅ 100% |
| **T9 (06/11/2026) — E-learning** | Thực thi WBS theo 5 thành viên: YOLO, OCR/Validate, UI/SQLite, Chương 1 & 6 | TV1, TV2, TV3, TV4, TV5 | [`compare_detectors.csv`](plate_app/reports/compare_detectors.csv), [`train_results.md`](plate_app/reports/train_results.md) | ✅ 100% |
| **T10 (13/11/2026) — E-learning** | Hoàn thiện Chương 1, Chương 2; sửa Best-Shot & xuất hình minh họa | TV1, TV2, TV3 | [`plate_app/reports/hinh_bao_cao/`](plate_app/reports/hinh_bao_cao), [`docs/GIAI_THICH.md`](docs/GIAI_THICH.md) | ✅ 100% |
| **T11 (20/11/2026) — E-learning** | Gán nhãn 143 biển, chạy `evaluate_ocr.py`, phân loại 22 lỗi YOLO, đo GPU/FPS & Chương 5 | TV4, TV1 | [`ocr_comparison.csv`](plate_app/reports/ocr_comparison.csv), [`detector_failures_analysis.csv`](plate_app/reports/detector_failures_analysis.csv) | ✅ 100% |
| **T12 (27/11/2026) — Hiện tại** | Rà soát toàn văn Word, hoàn thiện Chương 6, Slide/Minigame & đồng bộ GitHub nộp bài | Cả nhóm (TV1–TV5) | [`TIEN_DO_WBS.md`](TIEN_DO_WBS.md), [`requirements.txt`](requirements.txt) | 🔄 Đang hoàn thiện |
