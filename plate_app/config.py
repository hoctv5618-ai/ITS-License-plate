import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# Tính mọi đường dẫn từ vị trí file config.py để không phụ thuộc ổ đĩa cố định
BASE_DIR = Path(__file__).resolve().parent

# Tự động nhận diện data/raw hoặc thư mục dữ liệu thô cùng cấp
RAW_DIR = (
    BASE_DIR / "data" / "raw"
    if (BASE_DIR / "data" / "raw").exists()
    else BASE_DIR.parent / "ITS"
)

# Các thư mục và đường dẫn file chính trong ứng dụng
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "reports"
SPLIT_DIR = DATA_DIR / "processed" / "split"
RESULT_DIR = DATA_DIR / "results"

# Trỏ mặc định tới trọng số huấn luyện thật best_n640.pt
MODEL_PATH = MODEL_DIR / "best_n640.pt"
DATA_YAML_PATH = DATA_DIR / "data.yaml"
DB_PATH = DATA_DIR / "plates_history.db"
DB_BACKUP_PATH = DATA_DIR / "plates_history_backup.db"
TRAIN_RESULTS_MD = REPORT_DIR / "train_results.md"
OCR_SET_DIR = DATA_DIR / "ocr_set"
OCR_LABELS_CSV = DATA_DIR / "ocr_labels.csv"
OCR_COMPARISON_CSV = REPORT_DIR / "ocr_comparison.csv"
OCR_ABLATION_CSV = REPORT_DIR / "ocr_ablation.csv"
OCR_RAW_CACHE_CSV = REPORT_DIR / "ocr_raw_cache.csv"
OCR_ABLATION_PREPROCESS_CSV = REPORT_DIR / "ocr_ablation_preprocess.csv"

# Ngưỡng độ tự tin khi phát hiện biển số (YOLO), đánh giá (EVAL) và đọc ký tự (EasyOCR)
DETECT_CONF = 0.40
DETECT_IMGSZ = 640
EVAL_CONF = 0.25
EVAL_IOU = 0.50
OCR_CONF = 0.30
OCR_USE_GPU = True
OCR_ALLOWLIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZĐ-."

# Tỉ lệ nới rộng khung cắt (5% mỗi cạnh) để không cắt phạm vào viền chữ trên biển số
PADDING_RATIO = 0.05

# Chiều cao tối thiểu (pixel) của ảnh biển số: chỉ phóng to giữ nguyên tỉ lệ nếu nhỏ hơn 100px
PLATE_MIN_HEIGHT = 100

# Tham số thuật toán tiền xử lý ảnh (CLAHE và Gaussian Blur)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)
GAUSSIAN_KERNEL = (5, 5)

# Ngưỡng tỉ lệ Rộng/Cao (w/h = 2.1) phân biệt biển 2 dòng (< 2.1) và 1 dòng (>= 2.1)
TWO_LINE_RATIO = 2.10

# Cấu hình xử lý Video: xử lý mỗi N khung hình và bỏ phiếu số đông trên M khung hình
FRAME_SKIP = 3
VOTE_WINDOW = 5
VOTE_MAX_DIST = 80.0

# Số ký tự tối đa cho phép sửa nhầm chữ <-> số trong validate.py (giữ nguyên giá trị hiện tại = 3)
MAX_FIXED_CHARS = 3

# Các công tắc bật/tắt bước tiền xử lý ảnh và đọc 2 nửa được thêm sau (mặc định True, không đổi hành vi):
# 1. Ra đời từ clip TikTok & ảnh Honda City: xoay thẳng biển số theo góc đường thẳng Hough (Canny + HoughLinesP)
RULE_STRAIGHTEN = True
# 2. Ra đời từ ảnh xe Honda City trắng: cắt 4.5% hai mép trái/phải (và 4% trên/dưới) trên biển dài 1 dòng để bỏ viền kim loại
RULE_TRIM_EDGE = True
# 3. Ra đời từ clip TikTok biển 2 dòng nghiêng/mờ: tách đôi nửa trên (0..55%) và nửa dưới (45..100%) đọc lại khi lần đầu chưa hợp lệ
RULE_TWO_HALVES = True

# Các công tắc bật/tắt từng nhóm luật hậu xử lý được thêm sau từ các ca lỗi thực tế:
# 1. Ra đời từ clip TikTok (651f36827ca9c8e82188dd478a650b10.mp4): logo TikTok đè góc biển '60-AA 865.86' -> đọc ra '86585TKTOK', 'TKTOK'
RULE_STRIP_WATERMARK = True
# 2. Ra đời từ clip TikTok (651f36827ca9c8e82188dd478a650b10.mp4): biển 2 dòng nghiêng 15-20 độ bị đọc đảo dòng dưới lên trước ('865.8660-AA', '966.6460-B8', '865.86601')
RULE_SWAP_INVERTED_LINES = True
# 3. Ra đời từ ảnh xe Honda City trắng (hinh-anh-xe-honda-city-1024x768.jpg): viền kim loại phải của biển '51G-516.02' bị đọc thừa ký tự trong [17IJT] ở cuối ('616-516.027')
RULE_TRIM_TRAILING_BORDER = True
# 4. Ra đời từ clip TikTok (651f36827ca9c8e82188dd478a650b10.mp4): ánh xạ bổ sung EXTRA_TO_DIGIT = {'Q':'0', 'U':'0', 'J':'3', 'L':'4', 'T':'7'}
RULE_EXTRA_DIGIT_MAP = True

# Các mẫu dưới đây dựa trên định dạng biển phổ biến quan sát được, chưa đối chiếu văn bản quy định.
# Hạn chế: biển 4 số có 2 chữ cái, biển quân đội/ngoại giao/công vụ chưa được hỗ trợ.
PLATE_PATTERNS = [
    {"name": "oto_5so", "regex": r"^\d{2}[A-Z]-\d{3}\.\d{2}$", "mask": "DDLDDDDD"},
    {"name": "oto_4so", "regex": r"^\d{2}[A-Z]-\d{4}$", "mask": "DDLDDDD"},
    {"name": "xemay_5so", "regex": r"^\d{2}-[A-Z][A-Z1-9] \d{3}\.\d{2}$", "mask": "DDLDDDDDD"},
    {"name": "xemay_5so_2chu", "regex": r"^\d{2}-A[A-Z] \d{3}\.\d{2}$", "mask": "DDLLDDDDD"},
    {"name": "xemay_4so", "regex": r"^\d{2}-[A-Z][1-9] \d{4}$", "mask": "DDLDDDDD"},
]


def ensure_directories():
    """Tạo sẵn các thư mục chứa mô hình, dữ liệu, kết quả và báo cáo nếu chưa tồn tại."""
    for folder in [MODEL_DIR, DATA_DIR, REPORT_DIR, RESULT_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
    return True


if __name__ == "__main__":
    ensure_directories()
    print("=== KIỂM TRA CẤU HÌNH HỆ THỐNG (config.py) ===")
    print(f"- Thư mục ứng dụng (BASE_DIR) : {BASE_DIR}")
    print(f"- File trọng số YOLO mặc định : {MODEL_PATH}")
    print(f"- Số mẫu định dạng biển hỗ trợ: {len(PLATE_PATTERNS)}")
