# GHI CHÚ CHUẨN BỊ CHO GIAI ĐOẠN 3 (OCR & KIỂM TRA ĐỊNH DẠNG BIỂN SỐ)

Tài liệu này lưu lại các hạng mục được dời từ Giai đoạn 2 sang Giai đoạn 3 để đảm bảo không bỏ sót trong quá trình triển khai:

1. **Bộ nhãn chuẩn OCR (`data/ocr_set/` và `data/ocr_labels.csv`):**
   - Được sinh ra ở cuối Giai đoạn 2 bởi `tools/make_ocr_labelset.py` (150 ảnh biển số cắt từ tập `test`, phân tầng theo 3 yếu tố: `source` x `plate_type` x `height_bucket`, bỏ qua biển `< 15px`).
   - Chia thành 2 tập con qua cột `subset`: **`dev` (~2/3, khoảng 103 biển)** và **`final` (~1/3, khoảng 47 biển)**.
   - **Nguyên tắc chống rò rỉ khi tinh chỉnh (tuning):** Toàn bộ việc thử nghiệm và chọn tham số (chế độ tiền xử lý, luật sửa ký tự nhầm, cấu hình OCR) **chỉ được phép chạy trên tập `dev`**. Tập `final` chỉ được chạy **ĐÚNG MỘT LẦN DUY NHẤT** ở cuối cùng để lấy số liệu đưa vào báo cáo.
   - Hai cột `text` (chuỗi biển số đúng) và `readable` (`1` nếu mắt người đọc được, `0` nếu quá mờ/che khuất) do người dùng tự gõ tay qua `tools/label_tool.py`.

2. **Hàm `so_sanh_tien_xu_ly()` (Dời từ Giai đoạn 2 sang Giai đoạn 3):**
   - Chạy thực nghiệm EasyOCR trên tập `data/ocr_set/` (chỉ tính các ảnh có `readable == 1`) với **3 chế độ tiền xử lý**:
     1. **Không tiền xử lý (`raw`)**: đưa thẳng ảnh biển số cắt ra vào EasyOCR.
     2. **Xám + Resize giữ tỉ lệ + CLAHE (`use_threshold=False`)**: chế độ mặc định.
     3. **Xám + Resize giữ tỉ lệ + CLAHE + Gaussian + Otsu Threshold (`use_threshold=True`)**.
   - Xuất bảng so sánh vào `reports/preprocess_comparison.csv` (so sánh độ chính xác đọc đúng hoàn toàn Exact Match, độ chính xác ký tự Character Accuracy, tỉ lệ khớp Regex sau khi sửa ký tự nhầm, và thời gian xử lý trung bình).

3. **Quy tắc sửa ký tự nhầm trong `validate.py`:**
   - Làm riêng theo từng dạng biển số Việt Nam:
     - Ô tô 1 dòng (ví dụ: `30A-123.45`)
     - Ô tô 2 dòng vuông
     - Xe máy 2 dòng (ví dụ: `59-X1 123.45`, seri có thể là chữ + số hoặc 2 chữ)
     - Biển 4 số cũ (ví dụ: `29-L1 1234` hoặc `51N-4616`)
   - Chỉ áp dụng sửa ký tự (`0 ↔ O`, `1 ↔ I`, `8 ↔ B`, `5 ↔ S`...) khi chuỗi sau sửa khớp một dạng biển hợp lệ; nếu không khớp dạng nào thì giữ nguyên chuỗi gốc và đánh dấu `"không hợp lệ"` (không ép sửa).
   - Có ít nhất 15 ca kiểm thử đúng/sai trong `if __name__ == "__main__":` và in `PASS/FAIL`.

