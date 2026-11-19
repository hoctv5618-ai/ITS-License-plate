from datetime import datetime
from pathlib import Path
import cv2
import streamlit as st

from config import FRAME_SKIP, RESULT_DIR
from database import save_result
from detect import draw_detections
from image_io import save_image
from pipeline import process_video
from ui_image import shrink_for_display


def save_video_plate_record(video_name, idx, item):
    """Lưu khung hình đẹp nhất (best_frame) và ảnh biển cắt từ video vào đĩa rồi ghi SQLite."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = Path(video_name).stem
    img_p = RESULT_DIR / f"vidframe_{stamp}_{stem}_{idx}.jpg"
    crop_p = RESULT_DIR / f"vidcrop_{stamp}_{stem}_{idx}.jpg"
    save_image(item["best_frame"], img_p)
    save_image(item["crop_img"], crop_p)
    rec = dict(item)
    rec.update({"source_type": "video", "file_name": video_name, "image_path": str(img_p), "crop_path": str(crop_p)})
    return save_result(rec)


def render_video_summary_player(unique_plates, out_vid_path):
    """Hiển thị trình phát video kết quả (Edge AI Tracker) và các chỉ số tiết kiệm OCR."""
    if not (out_vid_path and Path(out_vid_path).exists() and Path(out_vid_path).stat().st_size > 0):
        return
    st.subheader("🎬 Video kết quả theo dõi đối tượng (Edge AI Tracker - ID # & Biển số)")
    v_col, d_col = st.columns([2.2, 1.0])
    v_col.video(str(out_vid_path))
    with d_col:
        total_seen = sum(x.get("seen_count", 1) for x in unique_plates)
        total_ocr = sum(x.get("ocr_reads", 1) for x in unique_plates)
        saved_pct = int(round((1.0 - total_ocr / max(1, total_seen)) * 100))
        st.metric("Tổng số xe / biển số duy nhất", len(unique_plates))
        st.metric("Số khung hình phát hiện / Số lần gọi OCR", f"{total_seen} / {total_ocr}")
        st.metric("Tiết kiệm OCR (Smart Best-Shot Trigger)", f"{max(0, saved_pct)}%")
        st.download_button("⬇️ Tải video kết quả (.webm)", data=Path(out_vid_path).read_bytes(), file_name=Path(out_vid_path).name, mime="video/webm")


def render_unique_video_plates(vname, unique_plates, auto_save, out_vid_path=None):
    """Hiển thị các khung hình Best-Shot và bảng tổng hợp từng xe (ID #)."""
    render_video_summary_player(unique_plates, out_vid_path)
    if not unique_plates:
        st.warning("Không phát hiện biển số nào đủ rõ trong toàn bộ video.")
        return
    table_rows = []
    for idx, item in enumerate(unique_plates, 1):
        tid = item.get("track_id", idx)
        st.divider()
        c1, c2 = st.columns([1.3, 1.0])
        drawn = draw_detections(item["best_frame"], [item["bbox"] + [item["det_conf"]]], [f"ID #{tid} | {item['text_clean']}"])
        c1.image(cv2.cvtColor(shrink_for_display(drawn), cv2.COLOR_BGR2RGB), caption=f"Best-Shot của ID #{tid} (Độ nét Laplacian: {item.get('sharpness', 0.0):.1f})")
        with c2:
            c2.image(cv2.cvtColor(item["crop_img"], cv2.COLOR_BGR2RGB), caption=f"Ảnh biển cắt (ID #{tid})")
            st.markdown(f"### `ID #{tid}` — `{item['text_clean']}` ({'✅ Hợp lệ' if item['is_valid'] else '⚠️ Không hợp lệ'})")
            st.write(f"- **Số frame theo dõi / Số lần chạy OCR:** `{item['seen_count']}` frame / `{item.get('ocr_reads', 1)}` lần")
            st.write(f"- **Độ tự tin YOLO / OCR:** `{item['det_conf']:.2f}` / `{item['ocr_conf']:.2f}` | **Loại:** `{item['plate_type'] or 'Chưa rõ'}`")
        table_rows.append({"track_id": f"ID #{tid}", "text_clean": item["text_clean"], "is_valid": item["is_valid"], "seen_frames": item["seen_count"], "ocr_reads": item.get("ocr_reads", 1), "sharpness": item.get("sharpness", 0.0), "plate_type": item["plate_type"], "province": item["province_code"], "det_conf": item["det_conf"], "ocr_conf": item["ocr_conf"]})
    st.subheader("📋 Bảng tổng hợp các phương tiện / biển số duy nhất")
    st.dataframe(table_rows, use_container_width=True)
    if not auto_save and st.button("💾 Lưu tất cả biển số từ Video vào CSDL", type="primary"):
        for idx, item in enumerate(unique_plates, 1):
            save_video_plate_record(vname, idx, item)
        st.success(f"Đã lưu {len(unique_plates)} biển số vào cơ sở dữ liệu!")


def render_video_tab(models, conf_thres, use_prep, auto_save):
    """Hiển thị tab tải video (mp4/avi/mov), chạy Edge AI Tracker + Smart OCR Trigger và xuất video."""
    up_vid = st.file_uploader("Tải lên file video giao thông (mp4, avi, mov):", type=["mp4", "avi", "mov"])
    skip_n = st.number_input("Xử lý mỗi N khung hình (FRAME_SKIP - khuyên dùng 2 hoặc 3):", min_value=1, max_value=30, value=FRAME_SKIP)
    if up_vid is None:
        st.info("Hãy tải lên một file video để bắt đầu theo dõi ID #, lọc Best-Shot và bỏ phiếu biển số.")
        return
    if st.button("▶️ Bắt đầu phân tích Video", type="primary"):
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        tmp_vid = RESULT_DIR / f"tmp_upload_{Path(up_vid.name).name}"
        out_vid = RESULT_DIR / f"tracked_{Path(up_vid.name).stem}.webm"
        tmp_vid.write_bytes(up_vid.getvalue())
        prog_bar = st.progress(0.0, text="Đang chạy Edge AI Tracker & Smart OCR Trigger trên video...")
        unique_list = process_video(tmp_vid, models, frame_skip=int(skip_n), progress_callback=lambda p: prog_bar.progress(p, text=f"Đang xử lý video: {int(p * 100)}%"), conf_thres=conf_thres, use_preprocess=use_prep, out_video_path=out_vid)
        prog_bar.progress(1.0, text="Hoàn tất phân tích và kết xuất video!")
        tmp_vid.unlink(missing_ok=True)
        if auto_save:
            for idx, item in enumerate(unique_list, 1):
                save_video_plate_record(up_vid.name, idx, item)
        st.session_state["last_video_name"] = up_vid.name
        st.session_state["last_video_plates"] = unique_list
        st.session_state["last_out_vid"] = str(out_vid)
    if "last_video_plates" in st.session_state:
        render_unique_video_plates(st.session_state.get("last_video_name", up_vid.name), st.session_state["last_video_plates"], auto_save, st.session_state.get("last_out_vid"))
