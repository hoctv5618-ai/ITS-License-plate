import importlib
from pathlib import Path
import subprocess
import sys
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import database
import detect
import preprocess
import validate
import ocr
import video_tracker
import pipeline
import ui_image
import ui_video
import ui_history

for _mod in (config, database, detect, preprocess, validate, ocr, video_tracker, pipeline, ui_image, ui_video, ui_history):
    importlib.reload(_mod)

from config import DETECT_CONF, MODEL_DIR, MODEL_PATH, ensure_directories
from database import init_db
from detect import load_yolo_model
from ocr import load_reader
from ui_history import render_history_tab
from ui_image import render_image_tab
from ui_video import render_video_tab

sys.stdout.reconfigure(encoding="utf-8")


@st.cache_resource
def load_cached_models(weight_path_str):
    """Tải mô hình YOLOv8 và EasyOCR một lần duy nhất vào bộ nhớ đệm của Streamlit.
    Đầu vào: chuỗi đường dẫn file trọng số YOLO. Đầu ra: dict {'yolo', 'reader'}.
    """
    yolo_model = load_yolo_model(weight_path_str)
    ocr_reader = load_reader()
    return {"yolo": yolo_model, "reader": ocr_reader}


def resolve_weight_path():
    """Kiểm tra file trọng số models/best_n640.pt; nếu chưa có thì trả về None kèm cảnh báo."""
    if MODEL_PATH.exists():
        return MODEL_PATH
    quick_pt = MODEL_DIR / "best_n640_quick.pt"
    if quick_pt.exists():
        st.warning(f"Đang tạm dùng '{quick_pt.name}'.")
        return quick_pt
    return None


def render_sidebar():
    """Vẽ thanh cấu hình bên trái (Sidebar): ngưỡng phát hiện, bật/tắt tiền xử lý và tự động lưu DB."""
    st.sidebar.header("⚙️ Cấu hình nhận diện")
    conf_thres = st.sidebar.slider("Ngưỡng độ tự tin YOLO (DETECT_CONF):", min_value=0.10, max_value=0.90, value=float(DETECT_CONF), step=0.05)
    use_prep = st.sidebar.checkbox("Bật tiền xử lý ảnh biển (Xám + Resize + CLAHE)", value=True)
    auto_save = st.sidebar.checkbox("Tự động lưu kết quả vào SQLite", value=True)
    st.sidebar.divider()
    st.sidebar.caption(f"Trọng số chuẩn: `{MODEL_PATH.name}`")
    return conf_thres, use_prep, auto_save


def main():
    """Hàm chính khởi tạo giao diện Streamlit gồm 3 tab: Ảnh | Video | Lịch sử."""
    st.set_page_config(page_title="Nhận diện biển số xe Việt Nam (ITS)", layout="wide")
    ensure_directories()
    init_db()
    st.title("🚗 Hệ thống nhận diện biển số xe Việt Nam trong ITS")
    conf_thres, use_prep, auto_save = render_sidebar()
    weight_p = resolve_weight_path()
    if weight_p is None:
        st.error(f"Không tìm thấy file trọng số mô hình tại `{MODEL_PATH}`! Hãy chạy `python train.py` trước.")
        models = {"yolo": None, "reader": None}
    else:
        models = load_cached_models(str(weight_p))
    tab_img, tab_vid, tab_hist = st.tabs(["🖼️ Ảnh", "🎥 Video", "📜 Lịch sử"])
    with tab_img:
        render_image_tab(models, conf_thres, use_prep, auto_save)
    with tab_vid:
        render_video_tab(models, conf_thres, use_prep, auto_save)
    with tab_hist:
        render_history_tab()


if __name__ == "__main__":
    if st.runtime.exists():
        main()
    else:
        sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]]))
