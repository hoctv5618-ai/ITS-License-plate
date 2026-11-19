# 🚗 Dự án Nhận diện Biển số Xe Việt Nam trong Hệ thống Giao thông Thông minh (ITS)

Mã nguồn chính, mô hình, cơ sở dữ liệu và tài liệu của ứng dụng nằm tại thư mục [`plate_app/`](plate_app/README.md).

## 1. Khởi chạy nhanh ứng dụng (Giao diện Web Streamlit)

```powershell
cd D:\ITS\plate_app
python -m streamlit run app.py
```

## 2. Cấu trúc thư mục gốc `D:\ITS`

| Thư mục / Tệp tin | Tác dụng |
|---|---|
| [`plate_app/`](plate_app/README.md) | **Thư mục ứng dụng chính:** Chứa toàn bộ mã nguồn Python, trọng số YOLOv8 (`models/best_n640.pt`), dữ liệu đã chia tập (`data/`), báo cáo thực nghiệm (`reports/`), công cụ phụ trợ (`tools/`) và tài liệu (`docs/`). |
| `ITS/` | **Dữ liệu thô ban đầu:** Chứa bộ ảnh gốc từ Kaggle (`kaggle_vnlp`), Roboflow (`roboflow_cuongta`) và thư mục dữ liệu không dùng (`unused_vehicles`). |
| `yolov8n.pt` | Trọng số YOLOv8n pretrained trên COCO dùng làm điểm khởi đầu khi huấn luyện. |
| `weights/` | Chứa trọng số dự phòng ngoài luồng chính (`yolo26n.pt` — không dùng trong ứng dụng). |
| `runs/` | Thư mục lưu kết quả kiểm thử tự sinh của Ultralytics khi chạy ngoài `plate_app/`. |

## 3. Tài liệu hướng dẫn & Báo cáo chi tiết

- 📘 **Hướng dẫn cài đặt, sử dụng giao diện, giải thích từng thư mục/tệp tin và lệnh thực thi:** Xem tại [`plate_app/README.md`](plate_app/README.md).
- 📊 **Báo cáo tổng hợp toàn diện dự án, quy trình 6 giai đoạn và kết quả thực nghiệm:** Xem tại [`plate_app/docs/BAO_CAO_DU_AN.md`](plate_app/docs/BAO_CAO_DU_AN.md).
- 🧠 **Giải thích chi tiết các thuật toán (kèm ví dụ số minh họa):** Xem tại [`plate_app/docs/GIAI_THICH.md`](plate_app/docs/GIAI_THICH.md).
- 📅 **Cấu trúc WBS & Đánh giá tiến độ chi tiết từng tuần (T8–T12):** Xem đầy đủ checklist tương tác tại [`TIEN_DO_WBS.md`](TIEN_DO_WBS.md).

## 4. Tóm tắt Tiến độ Thực hiện Theo Tuần (WBS Tracking)

> Cập nhật theo lịch E-learning làm việc nhóm — chi tiết checklist từng thành viên xem tại [`TIEN_DO_WBS.md`](TIEN_DO_WBS.md).

| Tuần & Mốc thời gian | Nội dung trọng tâm theo WBS | Sản phẩm minh chứng | Trạng thái |
| :--- | :--- | :--- | :---: |
| **T8 (30/10/2026)** | Chốt đề tài, WBS, làm sạch & chia tập 8.752 ảnh | [`plate_app/reports/source_stats.csv`](plate_app/reports/source_stats.csv) | ✅ 100% |
| **T9 (06/11/2026) — E-learning** | Thực thi WBS theo thành viên (YOLO, OCR/Validate, UI/SQLite) | [`plate_app/reports/compare_detectors.csv`](plate_app/reports/compare_detectors.csv) | ✅ 100% |
| **T10 (13/11/2026) — E-learning** | Hoàn thiện Chương 1 & Chương 2; sửa Best-Shot & xuất hình | [`plate_app/reports/hinh_bao_cao/`](plate_app/reports/hinh_bao_cao) | ✅ 100% |
| **T11 (20/11/2026) — E-learning** | Đo OCR, phân loại 22 lỗi YOLO, đo GPU/FPS & Chương 5 | [`plate_app/reports/ocr_comparison.csv`](plate_app/reports/ocr_comparison.csv) | 🔄 93% |
| **T12 (27/11/2026)** | Hoàn thiện Chương 6, rà soát cuối, Slide & nộp bài | `Bao_cao_tieu_luan_ITS.docx` | ⏳ 0% |
