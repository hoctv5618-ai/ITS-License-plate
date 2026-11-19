from collections import Counter
import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

# Tính đường dẫn từ vị trí file trong thư mục tools/
BASE_DIR = Path(__file__).resolve().parent.parent
ALL_DIR = BASE_DIR / "data" / "processed" / "all"
RAW_DIR = (
    BASE_DIR / "data" / "raw"
    if (BASE_DIR / "data" / "raw").exists()
    else BASE_DIR.parent / "hệ thống giao thông"
)


def collect_source_sizes_and_heights(all_dir, source_prefix):
    """Thu thập kích thước ảnh (w x h) và chiều cao biển số (pixel) của một nguồn.
    Khi YOLO thu phóng ảnh về imgsz (640 hoặc 960), cạnh dài nhất được đưa về đúng imgsz
    theo tỉ lệ scale = imgsz / max(w, h), nên chiều cao biển số cũng co giãn theo tỉ lệ này.
    """
    img_dir, lbl_dir = all_dir / "images", all_dir / "labels"
    size_counter = Counter()
    heights = {"orig": [], "640": [], "960": []}
    for img_path in sorted(img_dir.iterdir()):
        if not img_path.name.startswith(source_prefix):
            continue
        with Image.open(img_path) as im:
            img_w, img_h = im.size
        size_counter[f"{img_w}x{img_h}"] += 1
        max_side = float(max(img_w, img_h))
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        for line in lbl_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 5:
                h_orig = float(parts[4]) * img_h
                heights["orig"].append(h_orig)
                heights["640"].append(h_orig * (640.0 / max_side))
                heights["960"].append(h_orig * (960.0 / max_side))
    return size_counter, heights


def format_height_stats(height_list):
    """Tính trung vị và phần trăm biển số có chiều cao dưới 20px, 30px, 50px.
    Đầu vào: danh sách chiều cao biển số (pixel).
    Đầu ra: chuỗi định dạng kết quả thống kê.
    """
    if not height_list:
        return "Không có dữ liệu"
    total = len(height_list)
    med = float(np.median(height_list))
    u20 = sum(1 for h in height_list if h < 20.0) * 100.0 / total
    u30 = sum(1 for h in height_list if h < 30.0) * 100.0 / total
    u50 = sum(1 for h in height_list if h < 50.0) * 100.0 / total
    return f"Trung vị: {med:5.1f}px | <20px: {u20:5.2f}% | <30px: {u30:5.2f}% | <50px: {u50:5.2f}%"


def print_source_report(source_name):
    """In báo cáo phân bố kích thước ảnh và chiều cao biển số cho một nguồn dữ liệu.
    Đầu vào: source_name ('kaggle' hoặc 'roboflow').
    Đầu ra: Không có.
    """
    size_counter, heights = collect_source_sizes_and_heights(ALL_DIR, f"{source_name}_")
    total_imgs = sum(size_counter.values())
    pct_640 = size_counter.get("640x640", 0) * 100.0 / max(1, total_imgs)
    top_sizes = ", ".join(f"{k}: {v} ({v*100/total_imgs:.1f}%)" for k, v in size_counter.most_common(5))
    print(f"\n=== NGUỒN: {source_name.upper()} (Tổng {total_imgs} ảnh, {len(heights['orig'])} biển số) ===")
    print(f"- Số loại kích thước khác nhau : {len(size_counter)}")
    print(f"- 5 kích thước phổ biến nhất   : {top_sizes}")
    print(f"- Phần trăm ảnh đúng 640x640   : {pct_640:.2f}%")
    print(f"- Chiều cao biển (Ảnh gốc)     : {format_height_stats(heights['orig'])}")
    print(f"- Chiều cao biển (imgsz = 640) : {format_height_stats(heights['640'])}")
    print(f"- Chiều cao biển (imgsz = 960) : {format_height_stats(heights['960'])}")


def print_roboflow_readme():
    """Tìm và in nội dung mục pre-processing trong file README.roboflow.txt.
    Đầu vào: Không có. Đầu ra: Không có.
    """
    readmes = list((RAW_DIR / "roboflow_cuongta").rglob("README.roboflow.txt"))
    print("\n=== NỘI DUNG TIỀN XỬ LÝ TRONG README.roboflow.txt ===")
    if not readmes:
        print("Không tìm thấy file README.roboflow.txt.")
        return
    text = readmes[0].read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines()[19:]:
        print("  " + line)


if __name__ == "__main__":
    for src in ["kaggle", "roboflow"]:
        print_source_report(src)
    print_roboflow_readme()

