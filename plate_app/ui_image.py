import csv
from datetime import datetime
import io
import cv2
import numpy as np
import streamlit as st

from config import RESULT_DIR
from database import save_result
from detect import draw_detections
from image_io import save_image
from pipeline import process_image
from validate import validate_plate


def shrink_for_display(image, max_w=900):
    """Thu nhỏ ảnh quá lớn trước khi hiển thị lên giao diện để tránh tràn khung hình."""
    h, w = image.shape[:2]
    if w <= max_w:
        return image
    scale = max_w / float(w)
    return cv2.resize(image, (max_w, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def store_detection_to_db(file_name, item, orig_img, is_corrected=False, custom_text=None):
    """Lưu ảnh gốc, ảnh biển cắt ra đĩa (data/results/) và ghi bản ghi vào SQLite."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    img_p = RESULT_DIR / f"img_{stamp}_{file_name}"
    crop_p = RESULT_DIR / f"crop_{stamp}_{file_name}"
    save_image(orig_img, img_p)
    save_image(item["crop_img"], crop_p)
    rec = dict(item)
    if custom_text is not None:
        v = validate_plate(custom_text)
        rec.update({"text_clean": v["text_clean"] or custom_text.strip().upper(), "is_valid": v["is_valid"], "plate_type": v["plate_type"], "province_code": v["province_code"]})
    rec.update({"source_type": "image", "file_name": file_name, "is_corrected": is_corrected, "image_path": str(img_p), "crop_path": str(crop_p)})
    return save_result(rec)


def render_single_plate_card(fname, idx, item, orig_img):
    """Hiển thị thông tin chi tiết 1 biển số ở cột phải và ô sửa tay kèm nút Lưu."""
    c_a, c_b = st.columns(2)
    c_a.image(cv2.cvtColor(item["crop_img"], cv2.COLOR_BGR2RGB), caption="Ảnh biển cắt (lề 5%)")
    proc_vis = item["proc_img"]
    if proc_vis is not None and proc_vis.ndim == 3:
        proc_vis = cv2.cvtColor(proc_vis, cv2.COLOR_BGR2RGB)
    c_b.image(proc_vis, caption="Sau tiền xử lý", clamp=True)
    badge = "✅ Hợp lệ" if item["is_valid"] else "⚠️ Không hợp lệ"
    st.markdown(f"### Biển số: `{item['text_clean'] or '(Trống)'}` ({badge})")
    st.caption(
        f"Độ tự tin YOLO: `{item['det_conf']:.2f}` | Độ tự tin OCR: `{item['ocr_conf']:.2f}` | "
        f"Loại biển: `{item['plate_type'] or 'Chưa rõ'}` | Mã tỉnh: `{item['province_code'] or '--'}` | "
        f"Thời gian: `{item['process_ms']} ms` (Detect `{item['detect_ms']}ms`, Prep `{item['preprocess_ms']}ms`, OCR `{item['ocr_ms']}ms`)"
    )
    edit_key = f"edit_{fname}_{idx}"
    edited_txt = st.text_input("Sửa tay biển số (nếu OCR đọc nhầm):", value=item["text_clean"], key=edit_key)
    if st.button("Lưu bản sửa tay vào CSDL", key=f"btn_{fname}_{idx}"):
        store_detection_to_db(fname, item, orig_img, is_corrected=True, custom_text=edited_txt)
        st.success(f"Đã lưu biển '{edited_txt.strip().upper()}' (đánh dấu is_corrected=1) vào CSDL!")


def build_batch_csv_bytes(summary_rows):
    """Chuyển danh sách kết quả cả lô ảnh thành chuỗi byte CSV (UTF-8-SIG) để tải về."""
    buf = io.StringIO()
    if summary_rows:
        writer = csv.DictWriter(buf, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    return buf.getvalue().encode("utf-8-sig")


def render_image_tab(models, conf_thres, use_prep, auto_save):
    """Hiển thị tab xử lý nhiều ảnh tĩnh, chia 2 cột trực quan và xuất bảng tổng hợp cả lô."""
    uploads = st.file_uploader("Tải lên một hoặc nhiều ảnh xe (jpg, jpeg, png, webp):", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
    if not uploads:
        st.info("Hãy tải lên ít nhất 1 ảnh để bắt đầu nhận diện.")
        return
    summary_rows = []
    for up_file in uploads:
        st.divider()
        st.subheader(f"📄 File: {up_file.name}")
        raw_bytes = np.frombuffer(up_file.getvalue(), dtype=np.uint8)
        bgr_img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR) if raw_bytes.size > 0 else None
        if bgr_img is None:
            st.error(f"File '{up_file.name}' bị hỏng hoặc không phải định dạng ảnh hợp lệ.")
            continue
        plates = process_image(bgr_img, models, conf_thres=conf_thres, use_preprocess=use_prep)
        col_left, col_right = st.columns([1.2, 1.0])
        boxes_with_conf = [p["bbox"] + [p["det_conf"]] for p in plates]
        labels = [p["text_clean"] or "Plate" for p in plates]
        drawn = shrink_for_display(draw_detections(bgr_img, boxes_with_conf, labels))
        col_left.image(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB), caption=f"{up_file.name} (Phát hiện: {len(plates)} biển)")
        with col_right:
            if not plates:
                st.warning("Không phát hiện biển số nào trong bức ảnh này.")
            for idx, item in enumerate(plates, 1):
                render_single_plate_card(up_file.name, idx, item, bgr_img)
                if auto_save and f"saved_{up_file.name}_{idx}" not in st.session_state:
                    store_detection_to_db(up_file.name, item, bgr_img, is_corrected=False)
                    st.session_state[f"saved_{up_file.name}_{idx}"] = True
                summary_rows.append({
                    "file": up_file.name, "text_raw": item["text_raw"], "text_clean": item["text_clean"],
                    "is_valid": item["is_valid"], "plate_type": item["plate_type"], "province": item["province_code"],
                    "det_conf": item["det_conf"], "ocr_conf": item["ocr_conf"],
                    "detect_ms": item["detect_ms"], "preprocess_ms": item["preprocess_ms"], "ocr_ms": item["ocr_ms"], "total_ms": item["process_ms"],
                })
    if summary_rows:
        st.subheader("📊 Bảng tổng hợp kết quả lô ảnh")
        st.dataframe(summary_rows, use_container_width=True)
        st.download_button("⬇️ Tải bảng kết quả CSV", data=build_batch_csv_bytes(summary_rows), file_name="ket_qua_anh.csv", mime="text/csv")

