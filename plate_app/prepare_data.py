import csv
import shutil
import sys
from pathlib import Path
import cv2
import numpy as np

# Đổi chuẩn xuất màn hình sang UTF-8 để Windows Console không bị lỗi chữ tiếng Việt
sys.stdout.reconfigure(encoding="utf-8")

# Tính đường dẫn tương đối từ vị trí file; ưu tiên data/raw nếu đã chuyển sang D:\plate_app
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw" if (BASE_DIR / "data" / "raw").exists() else BASE_DIR.parent / "hệ thống giao thông"
SOURCES = {"kaggle": RAW_DIR / "kaggle_vnlp", "roboflow": RAW_DIR / "roboflow_cuongta"}
OUT_IMG_DIR = BASE_DIR / "data" / "processed" / "all" / "images"
OUT_LBL_DIR = BASE_DIR / "data" / "processed" / "all" / "labels"
REPORT_DIR = BASE_DIR / "reports"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
COORD_TOLERANCE = 0.05


def read_image_safe(image_path):
    """Đọc ảnh từ ổ đĩa hỗ trợ đường dẫn có dấu tiếng Việt trên Windows.
    Đầu vào: image_path (Path) - đường dẫn file ảnh.
    Đầu ra: mảng ảnh numpy (BGR) hoặc None nếu file ảnh bị lỗi.
    """
    # Dùng np.fromfile + imdecode vì cv2.imread bị lỗi với đường dẫn Unicode trên Windows
    try:
        raw_bytes = np.fromfile(str(image_path), dtype=np.uint8)
        return cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR) if raw_bytes.size > 0 else None
    except Exception:
        return None


def parse_yolo_label(label_path):
    """Kiểm tra file nhãn YOLO, ép class về 0 và kẹp tọa độ lệch nhẹ về [0, 1].
    Đầu vào: label_path (Path) - đường dẫn tới file nhãn .txt trong thư mục labels.
    Đầu ra: tuple (trạng_thái, danh_sách_dòng_nhãn_mới, danh_sách_w_h_chuẩn_hóa).
    """
    if not label_path.exists() or label_path.parent.name.lower() != "labels":
        return "missing_label", [], []
    lines = [s.strip() for s in label_path.read_text(encoding="utf-8", errors="ignore").splitlines() if s.strip()]
    if not lines:
        return "empty_label", [], []
    fixed_lines, norm_boxes = [], []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            return "invalid_format", [], []
        try:
            vals = [float(x) for x in parts[1:]]
        except ValueError:
            return "invalid_format", [], []
        if any(v < -COORD_TOLERANCE or v > 1.0 + COORD_TOLERANCE for v in vals) or vals[2] <= 0 or vals[3] <= 0:
            return "invalid_format", [], []
        # Kẹp các giá trị sát biên (do làm tròn số thực) về đúng đoạn [0.0, 1.0]
        cx, cy, w, h = [min(1.0, max(0.0, v)) for v in vals]
        fixed_lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        norm_boxes.append((w, h))
    return "ok", fixed_lines, norm_boxes


def build_source_summary(source_name, counts, box_counts, wh_ratios):
    """Tổng hợp các chỉ số thống kê của một nguồn dữ liệu thành từ điển.
    Đầu vào: tên nguồn, bộ đếm trạng thái, số box mỗi ảnh, danh sách tỉ lệ w/h.
    Đầu ra: dict chứa số liệu thống kê của nguồn đó.
    """
    # Tỉ lệ rộng/cao dưới 2.0 thường là biển số vuông 2 dòng (xe máy, ô tô vuông)
    under_2 = sum(1 for r in wh_ratios if r < 2.0)
    pct_under_2 = (under_2 * 100.0 / len(wh_ratios)) if wh_ratios else 0.0
    return {
        "source": source_name, "accepted_images": counts["ok"],
        "dropped_unreadable": counts["unreadable_image"], "dropped_missing_label": counts["missing_label"],
        "dropped_empty_label": counts["empty_label"], "dropped_invalid_format": counts["invalid_format"],
        "avg_boxes_per_image": round(float(np.mean(box_counts)) if box_counts else 0.0, 3),
        "median_box_wh_ratio": round(float(np.median(wh_ratios)) if wh_ratios else 0.0, 3),
        "pct_box_wh_under_2": round(pct_under_2, 2),
    }


def process_one_source(source_name, source_dir, dropped_records):
    """Duyệt một nguồn dữ liệu, lọc cặp ảnh-nhãn hợp lệ và copy sang processed/all.
    Đầu vào: tên nguồn, thư mục nguồn, danh sách ghi nhận file bị loại.
    Đầu ra: dict thống kê kết quả xử lý của nguồn.
    """
    counts = {"ok": 0, "unreadable_image": 0, "missing_label": 0, "empty_label": 0, "invalid_format": 0}
    box_counts, wh_ratios = [], []
    image_files = sorted(p for p in source_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    for img_path in image_files:
        img = read_image_safe(img_path)
        lbl_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
        status, fixed_lines, norm_boxes = ("unreadable_image", [], []) if img is None else parse_yolo_label(lbl_path)
        if status != "ok":
            counts[status] += 1
            dropped_records.append({"source": source_name, "file": str(img_path), "reason": status})
            continue
        counts["ok"] += 1
        new_stem = f"{source_name}_{counts['ok']:05d}"
        shutil.copy2(img_path, OUT_IMG_DIR / f"{new_stem}{img_path.suffix.lower()}")
        (OUT_LBL_DIR / f"{new_stem}.txt").write_text("\n".join(fixed_lines) + "\n", encoding="utf-8")
        img_h, img_w = img.shape[:2]
        box_counts.append(len(norm_boxes))
        wh_ratios.extend((bw * img_w) / max(1.0, bh * img_h) for bw, bh in norm_boxes)
    return build_source_summary(source_name, counts, box_counts, wh_ratios)


def prepare_all_data():
    """Hàm chính chuẩn bị toàn bộ dữ liệu từ các nguồn và lưu báo cáo CSV.
    Đầu vào: Không có (lấy cấu hình ở đầu file).
    Đầu ra: danh sách dict bảng thống kê theo từng nguồn.
    """
    for folder in [OUT_IMG_DIR, OUT_LBL_DIR, REPORT_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
    dropped_records = []
    stats_rows = [process_one_source(name, path, dropped_records) for name, path in SOURCES.items()]
    for path, fields, rows in [
        (REPORT_DIR / "dropped.csv", ["source", "file", "reason"], dropped_records),
        (REPORT_DIR / "source_stats.csv", list(stats_rows[0].keys()), stats_rows),
    ]:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print("\n=== BẢNG THỐNG KÊ DỮ LIỆU THEO NGUỒN ===")
    for row in stats_rows:
        print(row)
    return stats_rows


if __name__ == "__main__":
    prepare_all_data()
