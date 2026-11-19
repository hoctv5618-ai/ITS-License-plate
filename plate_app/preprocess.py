import random
import sys
import cv2
import numpy as np

from config import (
    CLAHE_CLIP_LIMIT,
    CLAHE_GRID_SIZE,
    GAUSSIAN_KERNEL,
    PADDING_RATIO,
    PLATE_MIN_HEIGHT,
    REPORT_DIR,
    RULE_STRAIGHTEN,
    RULE_TRIM_EDGE,
    SPLIT_DIR,
    TWO_LINE_RATIO,
)
from image_io import read_image, save_image

sys.stdout.reconfigure(encoding="utf-8")


def to_grayscale(image):
    """Chuyển ảnh màu BGR sang ảnh đa mức xám 1 kênh màu (0.299R + 0.587G + 0.114B)
    vì nhận dạng ký tự chỉ cần độ tương phản sáng-tối giữa chữ và nền, không cần màu sắc.
    """
    return image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def deskew_plate(gray_image, straighten=RULE_STRAIGHTEN, trim_edge=RULE_TRIM_EDGE):
    """Tự động xoay phẳng biển số dựa trên góc trung vị của các đường viền ngang (Canny + HoughLinesP)
    và cắt bỏ 4.5% viền kim loại hai bên đối với biển dài 1 dòng để tránh OCR đọc nhầm viền khung thành số 1, 7.
    """
    h, w = gray_image.shape[:2]
    out = gray_image
    if straighten:
        edges = cv2.Canny(cv2.GaussianBlur(gray_image, (3, 3), 0), 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=max(15, w // 4), minLineLength=int(w * 0.35), maxLineGap=10)
        ang = 0.0
        if lines is not None:
            angs = [float(np.degrees(np.arctan2(y2 - y1, x2 - x1))) for x1, y1, x2, y2 in lines.reshape(-1, 4)]
            valid_angs = [a for a in angs if -30.0 <= a <= 30.0]
            ang = float(np.median(valid_angs)) if valid_angs else 0.0
        if 1.5 <= abs(ang) <= 30.0:
            mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, 1.0)
            out = cv2.warpAffine(gray_image, mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    if trim_edge and (w / max(1.0, float(h))) >= TWO_LINE_RATIO and w >= 80:
        dx, dy = int(w * 0.045), int(h * 0.04)
        out = out[dy:h - dy, dx:w - dx]
    return out


def resize_plate(gray_image):
    """Phóng to ảnh biển số bằng nội suy Bicubic giữ nguyên tỉ lệ w/h nếu chiều cao < PLATE_MIN_HEIGHT."""
    h, w = gray_image.shape[:2]
    if h >= PLATE_MIN_HEIGHT:
        return gray_image
    new_w = max(1, int(round(w * (PLATE_MIN_HEIGHT / float(h)))))
    return cv2.resize(gray_image, (new_w, PLATE_MIN_HEIGHT), interpolation=cv2.INTER_CUBIC)


def apply_clahe(gray_image):
    """Cân bằng biểu đồ xám cục bộ (CLAHE) trên lưới 8x8 ô nhỏ để cứu chữ bị bóng râm hoặc lóa đèn."""
    return cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE).apply(gray_image)


def apply_threshold(gray_image):
    """Làm mờ Gaussian khử nhiễu hạt rồi tách nhị phân tự động bằng thuật toán Otsu."""
    blurred = cv2.GaussianBlur(gray_image, GAUSSIAN_KERNEL, 0)
    _, binary_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_img


def preprocess_plate(plate_image, use_threshold=False, straighten=RULE_STRAIGHTEN, trim_edge=RULE_TRIM_EDGE):
    """Thực hiện tuần tự chuỗi tiền xử lý ảnh biển số: Xám -> Xoay phẳng & Cắt viền -> Phóng to -> CLAHE."""
    if plate_image is None or plate_image.size == 0:
        return None
    gray = deskew_plate(to_grayscale(plate_image), straighten=straighten, trim_edge=trim_edge)
    enhanced = apply_clahe(resize_plate(gray))
    return apply_threshold(enhanced) if use_threshold else enhanced


def crop_from_yolo_label(image, label_line):
    """Cắt vùng biển số từ dòng nhãn YOLO kèm lề PADDING_RATIO (5%) mỗi phía, kẹp trong biên ảnh."""
    img_h, img_w = image.shape[:2]
    _, cx, cy, bw, bh = [float(x) for x in label_line.split()]
    w_px, h_px = bw * img_w, bh * img_h
    pad_x, pad_y = w_px * PADDING_RATIO, h_px * PADDING_RATIO
    x1, y1 = max(0, int(round((cx * img_w) - w_px / 2 - pad_x))), max(0, int(round((cy * img_h) - h_px / 2 - pad_y)))
    x2, y2 = min(img_w, int(round((cx * img_w) + w_px / 2 + pad_x))), min(img_h, int(round((cy * img_h) + h_px / 2 + pad_y)))
    return image[y1:y2, x1:x2]


def run_demo_on_test_set():
    """Thống kê chiều cao biển trên 200 mẫu test và ghép ảnh demo 4 bước cho 6 biển (3 vuông, 3 dài)."""
    rng = random.Random(42)
    sampled = rng.sample(sorted((SPLIT_DIR / "test" / "images").glob("*.*")), 200)
    heights, sq_crops, lg_crops = [], [], []
    for img_p in sampled:
        img = read_image(img_p)
        lbl_p = SPLIT_DIR / "test" / "labels" / f"{img_p.stem}.txt"
        for line in lbl_p.read_text(encoding="utf-8").splitlines():
            if len(line.split()) == 5:
                crop = crop_from_yolo_label(img, line)
                ch, cw = crop.shape[:2]
                heights.append(ch)
                if ch >= 30 and (cw / max(1, ch)) <= 1.5:
                    sq_crops.append(crop)
                elif ch >= 30 and (cw / max(1, ch)) >= 2.8:
                    lg_crops.append(crop)
    cols = []
    for crop in rng.sample(sq_crops, 3) + rng.sample(lg_crops, 3):
        g = resize_plate(deskew_plate(to_grayscale(crop)))
        c, t = apply_clahe(g), apply_threshold(apply_clahe(g))
        to_bgr = lambda im: cv2.resize(cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) if len(im.shape) == 2 else im, (g.shape[1], 100))
        cols.append(np.vstack([to_bgr(crop), to_bgr(g), to_bgr(c), to_bgr(t)]))
    save_image(np.hstack(cols), REPORT_DIR / "preprocess_demo.jpg")
    print(f"Đã lưu ảnh minh họa tiền xử lý tại: {REPORT_DIR / 'preprocess_demo.jpg'}")


if __name__ == "__main__":
    run_demo_on_test_set()
