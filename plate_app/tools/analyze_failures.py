"""Phân tích định lượng các ca lỗi phát hiện của best_n640.pt trên tập test.

Với mỗi ảnh trong reports/detector_failures/best_n640/, tính lại:
- Bỏ sót (FN): chiều cao biển, có chạm mép ảnh không, IoU tốt nhất với dự đoán.
- Báo nhầm (FP): độ tin cậy, IoU tốt nhất với nhãn thật (IoU 0.1-0.5 = khung lệch).
Kết quả: reports/detector_failures_analysis.csv
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import EVAL_CONF, EVAL_IOU, MODEL_DIR, REPORT_DIR, SPLIT_DIR  # noqa: E402
from detect import detect_plates, load_yolo_model  # noqa: E402
from evaluate_utils import compute_iou, get_trained_imgsz, load_gt_items, match_boxes_greedy  # noqa: E402
from image_io import read_image  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
EDGE_PX = 3


def main():
    model = load_yolo_model(MODEL_DIR / "best_n640.pt")
    imgsz = get_trained_imgsz(model)
    fail_dir = REPORT_DIR / "detector_failures" / "best_n640"
    rows = []
    for f in sorted(fail_dir.glob("*.jpg")):
        img_path = SPLIT_DIR / "test" / "images" / f.name
        lbl_path = SPLIT_DIR / "test" / "labels" / f"{img_path.stem}.txt"
        image = read_image(img_path)
        h_img, w_img = image.shape[:2]
        gts = load_gt_items(img_path, lbl_path)
        preds = detect_plates(image, model, conf_thres=EVAL_CONF, imgsz=imgsz)
        flags, extra = match_boxes_greedy([g["box"] for g in gts], preds, EVAL_IOU)
        for g, hit in zip(gts, flags):
            if hit:
                continue
            x1, y1, x2, y2 = g["box"]
            best = max([compute_iou(g["box"], p[:4]) for p in preds] or [0.0])
            edge = x1 <= EDGE_PX or y1 <= EDGE_PX or x2 >= w_img - EDGE_PX or y2 >= h_img - EDGE_PX
            rows.append({"file": f.name, "kind": "FN", "h_px": y2 - y1, "w_px": x2 - x1,
                         "touch_edge": int(edge), "best_iou": round(best, 3), "conf": "",
                         "n_gt": len(gts), "n_pred": len(preds)})
        for p in extra:
            x1, y1, x2, y2 = p[:4]
            best = max([compute_iou(g["box"], p[:4]) for g in gts] or [0.0])
            edge = x1 <= EDGE_PX or y1 <= EDGE_PX or x2 >= w_img - EDGE_PX or y2 >= h_img - EDGE_PX
            rows.append({"file": f.name, "kind": "FP", "h_px": y2 - y1, "w_px": x2 - x1,
                         "touch_edge": int(edge), "best_iou": round(best, 3),
                         "conf": round(float(p[4]), 3) if len(p) > 4 else "",
                         "n_gt": len(gts), "n_pred": len(preds)})
    out = REPORT_DIR / "detector_failures_analysis.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    print(f"Đã lưu: {out}")


if __name__ == "__main__":
    main()

