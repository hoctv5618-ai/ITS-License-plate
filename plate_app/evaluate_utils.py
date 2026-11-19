import sys
from pathlib import Path
import cv2
from PIL import Image

from config import EVAL_IOU, REPORT_DIR, SPLIT_DIR, TWO_LINE_RATIO
from image_io import read_image, save_image

sys.stdout.reconfigure(encoding="utf-8")


def compute_iou(box_a, box_b):
    """Tính tỉ lệ Giao trên Hợp (Intersection over Union - IoU) giữa hai hình chữ nhật.
    Thuật toán lấy diện tích phần giao nhau chia cho tổng diện tích vùng hợp của hai khung
    để đo mức độ trùng khít giữa khung dự đoán của YOLO và khung nhãn thật.
    """
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, float(inter_x2 - inter_x1))
    inter_h = max(0.0, float(inter_y2 - inter_y1))
    inter_area = inter_w * inter_h
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union_area = area_a + area_b - inter_area
    return (inter_area / union_area) if union_area > 0 else 0.0


def match_boxes_greedy(gt_boxes, pred_boxes, iou_thres=EVAL_IOU):
    """Ghép cặp từng khung nhãn thật với khung dự đoán chưa dùng có IoU lớn nhất (>= 0.5).
    Ví dụ: Ảnh có 1 biển thật G1 và 2 dự đoán P1 (IoU=0.8), P2 (IoU=0.6). G1 sẽ lấy P1 (IoU lớn nhất),
    đánh dấu P1 đã dùng; P2 còn dư không có biển thật nào ghép nên bị tính là 1 ca phát hiện nhầm (FP).
    Nếu G1 chỉ có P3 (IoU=0.3 < 0.5), G1 không ghép được nên tính là 1 ca bỏ sót (FN).
    """
    used_preds = set()
    gt_matched = []
    for gt in gt_boxes:
        best_iou, best_idx = 0.0, -1
        for idx, pred in enumerate(pred_boxes):
            if idx in used_preds:
                continue
            iou_val = compute_iou(gt, pred[:4])
            if iou_val > best_iou:
                best_iou, best_idx = iou_val, idx
        if best_idx != -1 and best_iou >= iou_thres:
            used_preds.add(best_idx)
            gt_matched.append(True)
        else:
            gt_matched.append(False)
    unmatched_preds = [p for i, p in enumerate(pred_boxes) if i not in used_preds]
    return gt_matched, unmatched_preds


def load_gt_items(img_path, lbl_path):
    """Đọc các biển số thật từ file nhãn và phân nhóm theo PIXEL ẢNH GỐC.
    Tính chiều cao trên ảnh gốc giúp các nhóm (<30px, 30-60px, >60px) cố định hoàn toàn,
    đảm bảo so sánh công bằng giữa mô hình train ở imgsz=640 và imgsz=960.
    """
    with Image.open(img_path) as im:
        img_w, img_h = im.size
    src = "kaggle" if img_path.name.startswith("kaggle_") else "roboflow"
    items = []
    for line in lbl_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = [float(x) for x in parts]
        w_px, h_px = bw * img_w, bh * img_h
        x1, y1 = int(round(cx * img_w - w_px / 2)), int(round(cy * img_h - h_px / 2))
        x2, y2 = int(round(cx * img_w + w_px / 2)), int(round(cy * img_h + h_px / 2))
        ptype = "square (<2.1)" if (w_px / max(1.0, h_px)) < TWO_LINE_RATIO else "long (>=2.1)"
        h_bucket = "<30px" if h_px < 30.0 else ("30-60px" if h_px <= 60.0 else ">60px")
        items.append({"box": [x1, y1, x2, y2], "source": src, "plate_type": ptype, "h_bucket": h_bucket})
    return items


def save_failure_image(image, gt_items, pred_boxes, out_path):
    """Vẽ khung thật màu XANH LÁ và khung dự đoán màu ĐỎ cho các ca bỏ sót hoặc báo nhầm."""
    drawn = image.copy()
    for item in gt_items:
        x1, y1, x2, y2 = item["box"]
        cv2.rectangle(drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for pred in pred_boxes:
        x1, y1, x2, y2 = pred[:4]
        cv2.rectangle(drawn, (x1, y1), (x2, y2), (0, 0, 255), 2)
    save_image(drawn, out_path)


def get_trained_imgsz(model, default_imgsz=640):
    """Đọc kích thước imgsz lúc huấn luyện được lưu bên trong file trọng số .pt của YOLO."""
    try:
        train_args = getattr(model, "ckpt", {}).get("train_args", {})
        return int(train_args.get("imgsz", default_imgsz))
    except Exception:
        return default_imgsz


if __name__ == "__main__":
    cx, cy, bw, bh, iw, ih = 0.5, 0.5, 0.2, 0.1, 1000, 500
    yolo_box = [int(round((cx - bw / 2) * iw)), int(round((cy - bh / 2) * ih)), int(round((cx + bw / 2) * iw)), int(round((cy + bh / 2) * ih))]
    cases = [
        ("IoU trùng hoàn toàn (=1.0)", abs(compute_iou([0, 0, 100, 100], [0, 0, 100, 100]) - 1.0) < 1e-6),
        ("IoU không giao nhau (=0.0)", abs(compute_iou([0, 0, 50, 50], [60, 60, 100, 100]) - 0.0) < 1e-6),
        ("IoU dịch nửa khung (=1/3)", abs(compute_iou([0, 0, 100, 100], [50, 0, 150, 100]) - 1 / 3) < 1e-6),
        ("IoU box nằm gọn trong box (=0.25)", abs(compute_iou([25, 25, 75, 75], [0, 0, 100, 100]) - 0.25) < 1e-6),
        ("Ghép cặp: 1 thật - 1 dự đoán đúng", match_boxes_greedy([[0, 0, 100, 100]], [[5, 5, 100, 100, 0.9]]) == ([True], [])),
        ("Ghép cặp: 1 thật - 2 dự đoán chồng nhau (1 FP)", match_boxes_greedy([[0, 0, 100, 100]], [[5, 0, 100, 100, 0.9], [10, 0, 100, 100, 0.8]]) == ([True], [[10, 0, 100, 100, 0.8]])),
        ("Ghép cặp: 2 thật gần nhau - 1 dự đoán (1 FN)", match_boxes_greedy([[0, 0, 100, 100], [10, 0, 110, 100]], [[2, 0, 102, 100, 0.9]]) == ([True, False], [])),
        ("Ghép cặp: không có dự đoán nào", match_boxes_greedy([[0, 0, 100, 100]], []) == ([False], [])),
        ("Ghép cặp: không có biển thật nào", match_boxes_greedy([], [[0, 0, 100, 100, 0.9]]) == ([], [[0, 0, 100, 100, 0.9]])),
        ("Ghép cặp: 2 biển thật nhận đúng cả 2", match_boxes_greedy([[0, 0, 50, 50], [200, 200, 250, 250]], [[2, 2, 50, 50, 0.9], [202, 202, 250, 250, 0.8]]) == ([True, True], [])),
        ("Đổi tọa độ YOLO chuẩn hóa sang pixel", yolo_box == [400, 225, 600, 275]),
    ]
    for idx, (name, ok) in enumerate(cases, 1):
        print(f"Ca {idx:02d} [{('PASS' if ok else 'FAIL')}]: {name}")
    print(f"=> Tổng kết: {sum(x[1] for x in cases)}/{len(cases)} ca PASS")

