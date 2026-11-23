"""Xuất Hình 5.1 (a, b) cho báo cáo từ các ca lỗi thật của best_n640.pt.
(a) kaggle_00320: bỏ sót các biển rất nhỏ ở xa (9-23 px) -> có ô phóng to.
(b) kaggle_00231: biển xe bên cạnh bị cắt ở mép ảnh, không có nhãn -> bị tính là báo nhầm.
"""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import EVAL_CONF, MODEL_DIR, REPORT_DIR, SPLIT_DIR  # noqa: E402
from detect import detect_plates, load_yolo_model  # noqa: E402
from evaluate_utils import get_trained_imgsz, load_gt_items  # noqa: E402
from image_io import read_image, save_image  # noqa: E402

OUT = REPORT_DIR / "hinh_bao_cao"
GREEN, RED = (0, 200, 0), (0, 0, 255)


def draw(name, model, imgsz):
    img_path = SPLIT_DIR / "test" / "images" / name
    img = read_image(img_path)
    gts = load_gt_items(img_path, SPLIT_DIR / "test" / "labels" / f"{img_path.stem}.txt")
    preds = detect_plates(img, model, conf_thres=EVAL_CONF, imgsz=imgsz)
    t = max(2, img.shape[1] // 400)
    for g in gts:
        x1, y1, x2, y2 = g["box"]
        cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, t)
    for p in preds:
        x1, y1, x2, y2 = p[:4]
        cv2.rectangle(img, (x1, y1), (x2, y2), RED, t)
    return img, gts


def legend(img):
    s = img.shape[1] / 1000.0
    h = int(46 * s)
    cv2.rectangle(img, (0, 0), (img.shape[1], h), (0, 0, 0), -1)
    cv2.putText(img, "Xanh la: nhan that   Do: du doan YOLO", (int(12 * s), int(32 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9 * s, (255, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)
    return img


def main():
    model = load_yolo_model(MODEL_DIR / "best_n640.pt")
    imgsz = get_trained_imgsz(model)

    a, gts = draw("kaggle_00320.jpg", model, imgsz)
    small = [g["box"] for g in gts if g["box"][3] - g["box"][1] < 30]
    if small:
        x1 = max(0, min(b[0] for b in small) - 40); y1 = max(0, min(b[1] for b in small) - 40)
        x2 = min(a.shape[1], max(b[2] for b in small) + 40); y2 = min(a.shape[0], max(b[3] for b in small) + 40)
        crop = a[y1:y2, x1:x2].copy()
        zw = a.shape[1] // 2
        zoom = cv2.resize(crop, (zw, int(zw * crop.shape[0] / crop.shape[1])), interpolation=cv2.INTER_CUBIC)
        zh = min(zoom.shape[0], a.shape[0] // 2)
        zoom = zoom[:zh]
        cv2.rectangle(a, (x1, y1), (x2, y2), (0, 255, 255), max(2, a.shape[1] // 400))
        oy = a.shape[0] - zh
        a[oy:, a.shape[1] - zw:] = zoom
        cv2.rectangle(a, (a.shape[1] - zw, oy), (a.shape[1] - 1, a.shape[0] - 1), (0, 255, 255), 4)
    save_image(legend(a), OUT / "Hinh_5_1a_loi_phat_hien_1.jpg")

    b, _ = draw("kaggle_00231.jpg", model, imgsz)
    save_image(legend(b), OUT / "Hinh_5_1b_loi_phat_hien_2.jpg")
    print("Đã lưu Hình 5.1a, 5.1b vào", OUT)


if __name__ == "__main__":
    main()

