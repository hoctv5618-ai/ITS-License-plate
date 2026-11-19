import csv
import sys
from pathlib import Path
from PIL import Image
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import OCR_LABELS_CSV, OCR_SET_DIR

FIELDS = ["file", "source", "plate_type", "height_px", "height_bucket", "subset", "text", "readable", "excluded"]


def load_rows():
    """Đọc toàn bộ danh sách 150 biển số từ file data/ocr_labels.csv với mã hóa UTF-8.
    Đầu vào: Không có. Đầu ra: Danh sách các dict biểu diễn từng dòng trong CSV.
    """
    if not OCR_LABELS_CSV.exists():
        return []
    with open(OCR_LABELS_CSV, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save_rows(rows):
    """Ghi đè an toàn toàn bộ danh sách dòng vào data/ocr_labels.csv theo đúng thứ tự cột FIELDS.
    Giữ nguyên khóa cột subset và các cột thông tin khác để không bị mất dữ liệu khi lưu.
    """
    with open(OCR_LABELS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in FIELDS})


def find_first_unlabeled(rows):
    """Tìm chỉ số của biển số đầu tiên không bị loại trừ (excluded != 1) và chưa được gán nhãn (readable trống)."""
    for idx, r in enumerate(rows):
        if str(r.get("excluded", "0")).strip() == "1":
            continue
        if str(r.get("readable", "")).strip() == "":
            return idx
    for idx, r in enumerate(rows):
        if str(r.get("excluded", "0")).strip() != "1":
            return idx
    return 0


def render_plate_preview(row, zoom_factor):
    """Hiển thị ảnh biển số được phóng to theo hệ số 1x/2x/3x và thông tin chi tiết của ảnh."""
    img_path = OCR_SET_DIR / row["file"]
    if not img_path.exists():
        st.error(f"Không tìm thấy ảnh: {img_path}")
        return
    with Image.open(img_path) as im:
        w, h = im.size
        resized = im.resize((w * zoom_factor, h * zoom_factor), Image.Resampling.BICUBIC)
        st.image(resized, caption=f"{row['file']} ({w}x{h} px -> phóng {zoom_factor}x)")
    st.markdown(
        f"**File:** `{row['file']}` | **Nguồn:** `{row['source']}` | "
        f"**Loại:** `{row['plate_type']}` | **Cao gốc:** `{row['height_px']}px ({row['height_bucket']})` | "
        f"**Tập:** `{row.get('subset', 'dev')}`"
    )


def main():
    """Hàm chính dựng giao diện gán nhãn OCR nhanh trên trình duyệt bằng Streamlit."""
    st.set_page_config(page_title="Công cụ gán nhãn OCR biển số", layout="centered")
    st.title("Công cụ gán nhãn bộ kiểm thử OCR (150 biển)")
    rows = load_rows()
    if not rows:
        st.warning("Chưa có file data/ocr_labels.csv. Hãy chạy tools/make_ocr_labelset.py trước.")
        return
    if "cur_idx" not in st.session_state:
        st.session_state.cur_idx = find_first_unlabeled(rows)
    valid_indices = [i for i, r in enumerate(rows) if str(r.get("excluded", "0")).strip() != "1"]
    if not valid_indices:
        st.warning("Tất cả các dòng đều có excluded = 1.")
        return
    if st.session_state.cur_idx not in valid_indices:
        st.session_state.cur_idx = find_first_unlabeled(rows)
    dev_rows = [rows[i] for i in valid_indices if rows[i].get("subset", "dev") == "dev"]
    fin_rows = [rows[i] for i in valid_indices if rows[i].get("subset", "dev") == "final"]
    dev_done = sum(1 for r in dev_rows if str(r.get("readable", "")).strip() != "")
    fin_done = sum(1 for r in fin_rows if str(r.get("readable", "")).strip() != "")
    dev_rem = len(dev_rows) - dev_done
    fin_rem = len(fin_rows) - fin_done
    done_cnt = dev_done + fin_done
    total_valid = len(valid_indices)
    st.progress(
        done_cnt / max(1, total_valid),
        text=(
            f"Tiến độ: Đã gán {done_cnt}/{total_valid} biển | "
            f"Còn phải gán: dev còn {dev_rem}/{len(dev_rows)}, final còn {fin_rem}/{len(fin_rows)} "
            f"(đã bỏ qua {len(rows) - total_valid} biển excluded=1)"
        ),
    )
    pos = valid_indices.index(st.session_state.cur_idx)
    idx = valid_indices[pos]
    row = rows[idx]
    zoom = st.radio("Mức phóng to ảnh biển số:", [1, 2, 3], index=1, horizontal=True, format_func=lambda x: f"{x}x")
    render_plate_preview(row, zoom)
    new_text = st.text_input("Nhập chuỗi biển số (ví dụ 30A-123.45 hoặc 59-X1 123.45):", value=row.get("text", ""), key=f"in_{idx}")
    c1, c2, c3 = st.columns(3)
    if c1.button("Lưu và tiếp", type="primary", use_container_width=True):
        rows[idx]["text"], rows[idx]["readable"] = new_text.strip().upper(), "1"
        save_rows(rows)
        st.session_state.cur_idx = valid_indices[min(pos + 1, len(valid_indices) - 1)]
        st.rerun()
    if c2.button("Không đọc được (readable=0)", use_container_width=True):
        rows[idx]["text"], rows[idx]["readable"] = "", "0"
        save_rows(rows)
        st.session_state.cur_idx = valid_indices[min(pos + 1, len(valid_indices) - 1)]
        st.rerun()
    if c3.button("Quay lại", use_container_width=True) and pos > 0:
        st.session_state.cur_idx = valid_indices[pos - 1]
        st.rerun()


if __name__ == "__main__":
    if st.runtime.exists():
        main()
    else:
        import subprocess
        sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())]))

