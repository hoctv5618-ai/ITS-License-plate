import csv
import io
from pathlib import Path
import cv2
import streamlit as st

from database import backup_and_clear_history, delete_record, get_stats, load_history
from image_io import read_image


def render_statistics_cards():
    """Hiển thị 4 thẻ thống kê tổng quan: tổng lượt, tỉ lệ hợp lệ, top mã tỉnh và thời gian TB."""
    stats = get_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng lượt nhận diện", f"{stats['total']}")
    c2.metric("Tỉ lệ biển hợp lệ", f"{stats['valid_pct']:.1f}%")
    top_str = ", ".join(f"{p} ({n})" for p, n in stats["top_provinces"][:3]) or "Chưa có"
    c3.metric("Top mã tỉnh", top_str)
    c4.metric("Thời gian xử lý TB", f"{stats['avg_ms']:.1f} ms")


def export_history_csv(rows):
    """Chuyển danh sách bản ghi lịch sử thành byte CSV (UTF-8-SIG) để tải xuống."""
    buf = io.StringIO()
    if rows:
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")


def render_detail_and_delete(rows):
    """Cho phép chọn 1 bản ghi theo ID để xem lại ảnh gốc + ảnh biển cắt và xóa bản ghi."""
    if not rows:
        return
    st.subheader("🔍 Xem ảnh chi tiết & Quản lý bản ghi")
    options = {r["id"]: f"ID #{r['id']} | {r['timestamp']} | {r['text_clean'] or '(Trống)'} ({r['file_name']})" for r in rows}
    sel_id = st.selectbox("Chọn bản ghi cần xem hoặc xóa:", options=list(options.keys()), format_func=lambda k: options[k])
    rec = next((r for r in rows if r["id"] == sel_id), None)
    if rec is None:
        return
    col1, col2 = st.columns(2)
    img_p, crop_p = Path(rec.get("image_path", "")), Path(rec.get("crop_path", ""))
    if img_p.exists() and img_p.is_file():
        img_bgr = read_image(img_p)
        if img_bgr is not None:
            col1.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption=f"Ảnh gốc ({img_p.name})")
    else:
        col1.info("Không có file ảnh gốc lưu kèm.")
    if crop_p.exists() and crop_p.is_file():
        crop_bgr = read_image(crop_p)
        if crop_bgr is not None:
            col2.image(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB), caption=f"Ảnh biển cắt ({crop_p.name})")
    if st.button(f"🗑️ Xóa bản ghi ID #{sel_id}", type="secondary"):
        if delete_record(sel_id):
            st.success(f"Đã xóa bản ghi ID #{sel_id}!")
            st.rerun()


def render_history_tab():
    """Hiển thị tab Lịch sử: khối thống kê, bộ lọc tìm kiếm, bảng dữ liệu, xuất CSV và xem/xóa."""
    render_statistics_cards()
    if st.button("🧹 Sao lưu & Xóa dữ liệu thử trong CSDL (lưu ra plates_history_backup.db)", type="secondary"):
        bak_file, deleted_cnt = backup_and_clear_history()
        st.success(f"Đã sao lưu ra `{bak_file}` và xóa {deleted_cnt} bản ghi thử nghiệm!")
        st.rerun()
    st.divider()
    f1, f2, f3, f4 = st.columns(4)
    q_plate = f1.text_input("Tìm theo biển số:", placeholder="VD: 30A hoặc 59-X1")
    q_date = f2.text_input("Lọc theo ngày (YYYY-MM-DD):", placeholder="VD: 2026-10-30")
    q_src = f3.selectbox("Nguồn:", ["all", "image", "video"], format_func=lambda x: {"all": "Tất cả", "image": "Ảnh tĩnh", "video": "Video"}[x])
    q_val = f4.selectbox("Trạng thái:", ["all", 1, 0], format_func=lambda x: {"all": "Tất cả", 1: "Hợp lệ", 0: "Không hợp lệ"}[x])
    rows = load_history({"plate_query": q_plate, "date_str": q_date.strip(), "source_type": q_src, "is_valid": q_val})
    if not rows:
        st.info("Không có bản ghi nào khớp với bộ lọc hiện tại.")
        return
    st.dataframe(rows, use_container_width=True)
    st.download_button("⬇️ Xuất lịch sử ra CSV", data=export_history_csv(rows), file_name="lich_su_bien_so.csv", mime="text/csv")
    render_detail_and_delete(rows)

