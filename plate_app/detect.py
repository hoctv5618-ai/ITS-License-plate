import sys
from pathlib import Path
import cv2

from config import DETECT_CONF, DETECT_IMGSZ, MODEL_PATH, PADDING_RATIO, REPORT_DIR, SPLIT_DIR
from image_io import read_image, save_image

sys.stdout.reconfigure(encoding="utf-8")


def load_yolo_model(model_path=MODEL_PATH):
    """Tải mô hình YOLO từ file trọng số .pt để phát hiện biển số xe.
    Đầu vào: model_path (Path/str) - đường dẫn tới file trọng số (mặc định models/best.pt).
    Đầu ra: đối tượng YOLO đã tải, hoặc None nếu thiếu file hay chưa cài ultralytics.
    """
    path_obj = Path(model_path)
    if not path_obj.exists():
        print(f"[LỖI] Không tìm thấy file mô hình tại: {path_obj}")
        print("Hướng dẫn: Hãy chạy 'python train.py' để huấn luyện hoặc đặt file best.pt vào thư mục models/.")
        return None
    try:
        from ultralytics import YOLO
        return YOLO(str(path_obj))
    except ImportError:
        print("[LỖI] Chưa cài thư viện ultralytics. Hãy chạy: pip install ultralytics")
        return None


def detect_plates(image, model, conf_thres=DETECT_CONF, imgsz=DETECT_IMGSZ):
    """Phát hiện tọa độ các biển số xe trong một bức ảnh bằng mô hình YOLO.
    Thuật toán chia ảnh thành lưới ô vuông theo kích thước imgsz, dự đoán tọa độ khung bao
    và lọc bỏ các khung chồng lấp bằng Non-Maximum Suppression (NMS).
    """
    if image is None or image.size == 0 or model is None:
        return []
    # Truyền đúng imgsz (640 hoặc 960) tương ứng với cấu hình lúc huấn luyện mô hình
    results = model.predict(source=image, conf=conf_thres, imgsz=imgsz, verbose=False)
    if not results or results[0].boxes is None:
        return []
    boxes_list = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0].item())
        boxes_list.append([int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)), conf])
    return boxes_list


def crop_plate(image, box, pad_ratio=PADDING_RATIO):
    """Cắt vùng ảnh biển số từ khung bao [x1, y1, x2, y2] kèm mở rộng lề 5% mỗi phía.
    Đầu vào: image (ảnh BGR), box (list 4 hoặc 5 số), pad_ratio (tỉ lệ lề, mặc định 0.05).
    Đầu ra: mảng ảnh vùng biển số đã cắt, hoặc None nếu tọa độ không hợp lệ.
    """
    if image is None or len(box) < 4:
        return None
    img_h, img_w = image.shape[:2]
    x1, y1, x2, y2 = box[:4]
    w_box, h_box = max(1, x2 - x1), max(1, y2 - y1)
    # Thêm lề 5% xung quanh để không cắt phạm vào nét chữ sát viền biển số
    pad_x, pad_y = int(round(w_box * pad_ratio)), int(round(h_box * pad_ratio))
    cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    cx2, cy2 = min(img_w, x2 + pad_x), min(img_h, y2 + pad_y)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return image[cy1:cy2, cx1:cx2].copy()


def draw_detections(image, boxes, labels=None):
    """Vẽ khung chữ nhật màu xanh lá, chuỗi biển số (nếu có) và độ tự tin lên bản sao ảnh gốc.
    Đầu vào: image (mảng ảnh BGR), boxes (danh sách [x1, y1, x2, y2, conf]), labels (danh sách chuỗi).
    Đầu ra: mảng ảnh BGR mới đã được vẽ khung kết quả.
    """
    if image is None:
        return None
    drawn = image.copy()
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box[:4]
        conf = box[4] if len(box) >= 5 else 1.0
        cv2.rectangle(drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)
        txt = labels[idx] if (labels and idx < len(labels) and labels[idx]) else "Plate"
        label = f"{txt} ({conf:.2f})"
        cv2.putText(drawn, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return drawn


if __name__ == "__main__":
    yolo_model = load_yolo_model()
    test_images = sorted((SPLIT_DIR / "test" / "images").glob("*.*"))
    if yolo_model is not None and test_images:
        sample_img = read_image(test_images[0])
        found_boxes = detect_plates(sample_img, yolo_model)
        out_img = draw_detections(sample_img, found_boxes)
        save_image(out_img, REPORT_DIR / "detect_demo.jpg")
        print(f"Phát hiện được {len(found_boxes)} biển số trên ảnh {test_images[0].name}.")

