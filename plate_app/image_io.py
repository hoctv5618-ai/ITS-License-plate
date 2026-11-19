import sys
from pathlib import Path
import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


def read_image(image_path):
    """Đọc ảnh màu BGR từ ổ đĩa, an toàn với đường dẫn có dấu tiếng Việt trên Windows.
    Thuật toán dùng np.fromfile đọc mảng byte thô rồi cv2.imdecode giải mã thành ảnh,
    vì hàm cv2.imread mặc định dùng C++ cũ sẽ trả về None khi đường dẫn có dấu Unicode.
    """
    path_obj = Path(image_path)
    if not path_obj.exists() or not path_obj.is_file():
        return None
    try:
        raw_bytes = np.fromfile(str(path_obj), dtype=np.uint8)
        if raw_bytes.size == 0:
            return None
        return cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    except Exception:
        return None


def save_image(image, output_path):
    """Lưu mảng ảnh numpy ra ổ đĩa, hỗ trợ đường dẫn có dấu tiếng Việt trên Windows.
    Thuật toán dùng cv2.imencode nén ma trận ảnh thành mảng byte theo đuôi file (.jpg/.png),
    sau đó gọi .tofile() để ghi trực tiếp xuống đĩa thay cho cv2.imwrite.
    """
    if image is None or image.size == 0:
        return False
    out_obj = Path(output_path)
    out_obj.parent.mkdir(parents=True, exist_ok=True)
    ext = out_obj.suffix.lower() if out_obj.suffix else ".jpg"
    success, encoded_bytes = cv2.imencode(ext, image)
    if not success:
        return False
    encoded_bytes.tofile(str(out_obj))
    return True


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    test_imgs = sorted((base_dir / "data" / "processed" / "split" / "test" / "images").glob("*.jpg"))
    if not test_imgs:
        print("Không tìm thấy ảnh test để chạy thử image_io.py.")
    else:
        sample_img = read_image(test_imgs[0])
        out_test = base_dir / "reports" / "io_test_sample.jpg"
        ok = save_image(sample_img, out_test)
        print(f"Đọc ảnh mẫu : {test_imgs[0].name} -> Kích thước: {sample_img.shape}")
        print(f"Ghi ảnh thử  : {out_test} -> Thành công: {ok}")

