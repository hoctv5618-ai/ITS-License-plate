import csv
import sys
from pathlib import Path

from config import DATA_YAML_PATH, EVAL_CONF, EVAL_IOU, MODEL_DIR, REPORT_DIR, SPLIT_DIR
from detect import detect_plates, load_yolo_model
from evaluate_utils import get_trained_imgsz, load_gt_items, match_boxes_greedy, save_failure_image
from image_io import read_image

sys.stdout.reconfigure(encoding="utf-8")

# Danh sách các file trọng số cần đánh giá và đặt cạnh nhau trong compare_detectors.csv
WEIGHT_CANDIDATES = [
    MODEL_DIR / "best_n640.pt",
    MODEL_DIR / "best_n960.pt",
    MODEL_DIR / "best_n640_quick.pt",
    MODEL_DIR / "best.pt",
]
COMPARE_CSV = REPORT_DIR / "compare_detectors.csv"


def format_recall_row(label_name, records):
    """Tính Recall (%), tổng số biển của một nhóm và gắn cảnh báo nếu dưới 30 biển."""
    total = len(records)
    hits = sum(1 for r in records if r["hit"])
    rec_pct = (hits * 100.0 / total) if total > 0 else 0.0
    note = " [CẢNH BÁO: <30 biển, số liệu không đáng tin]" if total < 30 else ""
    print(f"  - {label_name:<22} | Số biển: {total:4d} | Phát hiện: {hits:4d} | Recall: {rec_pct:6.2f}%{note}")
    return round(rec_pct, 2)


def print_detailed_recall_tables(model_name, train_imgsz, records):
    """In bảng Recall tách theo loại biển, nhóm chiều cao ảnh gốc, nguồn và bảng chéo."""
    print(f"\n=== BẢNG RECALL CHI TIẾT: {model_name} (imgsz={train_imgsz}, EVAL_CONF={EVAL_CONF}, IoU={EVAL_IOU}) ===")
    print("a) Theo loại biển (tỉ lệ w/h):")
    for ptype in ["square (<2.1)", "long (>=2.1)"]:
        format_recall_row(ptype, [r for r in records if r["plate_type"] == ptype])
    print("b) Theo chiều cao biển trên PIXEL ẢNH GỐC:")
    h_recalls = {}
    for hb in ["<30px", "30-60px", ">60px"]:
        h_recalls[hb] = format_recall_row(hb, [r for r in records if r["h_bucket"] == hb])
    print("c) Theo nguồn và BẢNG CHÉO (Chiều cao ảnh gốc x Nguồn):")
    for src in ["kaggle", "roboflow"]:
        format_recall_row(f"Nguồn {src} (tổng)", [r for r in records if r["source"] == src])
        for hb in ["<30px", "30-60px", ">60px"]:
            sub = [r for r in records if r["source"] == src and r["h_bucket"] == hb]
            format_recall_row(f"  {src} x {hb}", sub)
    return h_recalls


def scan_test_set_predictions(model, train_imgsz, fail_dir):
    """Quét từng ảnh tập test ở đúng imgsz huấn luyện, ghép cặp IoU và lưu ảnh lỗi."""
    fail_dir.mkdir(parents=True, exist_ok=True)
    test_imgs = sorted((SPLIT_DIR / "test" / "images").glob("*.*"))
    records, fail_count = [], 0
    for img_path in test_imgs:
        lbl_path = SPLIT_DIR / "test" / "labels" / f"{img_path.stem}.txt"
        gt_items = load_gt_items(img_path, lbl_path)
        image = read_image(img_path)
        preds = detect_plates(image, model, conf_thres=EVAL_CONF, imgsz=train_imgsz)
        matched_flags, extra_preds = match_boxes_greedy([g["box"] for g in gt_items], preds, EVAL_IOU)
        for item, is_hit in zip(gt_items, matched_flags):
            records.append({**item, "hit": is_hit})
        # Lưu ảnh nếu có bất kỳ biển thật nào bị bỏ sót hoặc có khung dự đoán nhầm
        if not all(matched_flags) or len(extra_preds) > 0:
            fail_count += 1
            save_failure_image(image, gt_items, preds, fail_dir / img_path.name)
    print(f"Đã lưu {fail_count} ảnh bỏ sót/báo nhầm vào: {fail_dir}")
    return records


def evaluate_one_model(weight_path):
    """Đánh giá toàn diện một file trọng số YOLO và trả về dòng kết quả tổng hợp."""
    model = load_yolo_model(weight_path)
    if model is None:
        return None
    train_imgsz = get_trained_imgsz(model)
    val_res = model.val(data=str(DATA_YAML_PATH), split="test", imgsz=train_imgsz, verbose=False)
    fail_dir = REPORT_DIR / "detector_failures" / weight_path.stem
    records = scan_test_set_predictions(model, train_imgsz, fail_dir)
    h_recalls = print_detailed_recall_tables(weight_path.name, train_imgsz, records)
    return {
        "model_file": weight_path.name,
        "train_imgsz": train_imgsz,
        "precision": round(float(val_res.box.mp), 4),
        "recall": round(float(val_res.box.mr), 4),
        "map50": round(float(val_res.box.map50), 4),
        "map50_95": round(float(val_res.box.map), 4),
        "recall_u30": h_recalls["<30px"],
        "recall_30_60": h_recalls["30-60px"],
        "recall_o60": h_recalls[">60px"],
        "speed_ms": round(float(val_res.speed.get("inference", 0.0)), 2),
    }


def compare_detectors():
    """Chạy đánh giá lần lượt các file trọng số hiện có và xuất compare_detectors.csv."""
    existing_weights = [p for p in WEIGHT_CANDIDATES if p.exists()]
    if not existing_weights:
        print("[THÔNG BÁO] Chưa có file trọng số nào trong models/ để đánh giá.")
        print("Hướng dẫn: Hãy chạy train.py tạo models/best_n640.pt và models/best_n960.pt trước.")
        return []
    rows = [evaluate_one_model(w) for w in existing_weights]
    rows = [r for r in rows if r is not None]
    if rows:
        COMPARE_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(COMPARE_CSV, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nĐã xuất bảng so sánh các mô hình tại: {COMPARE_CSV}")
    return rows


if __name__ == "__main__":
    compare_detectors()

