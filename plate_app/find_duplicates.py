import csv
import hashlib
import random
import sys
from pathlib import Path
import cv2
import imagehash
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

# Tham số cấu hình đặt ở đầu file, tính đường dẫn từ vị trí script
BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "data" / "processed" / "all" / "images"
REPORT_CSV = BASE_DIR / "reports" / "duplicates.csv"
SAMPLE_DIR = BASE_DIR / "reports" / "duplicate_samples"
PHASH_THRESHOLD = 6
SAMPLE_COUNT = 10
PROGRESS_STEP = 500


def compute_hashes(image_path):
    """Tính mã băm MD5 (trùng từng byte) và số nguyên pHash 64-bit (gần trùng).
    Thuật toán pHash thu nhỏ ảnh, biến đổi tần số DCT để giữ cấu trúc sáng tối chung,
    giúp phát hiện hai ảnh giống nhau dù bị nén JPEG lại hoặc đổi kích thước.
    """
    md5_str = hashlib.md5(image_path.read_bytes()).hexdigest()
    with Image.open(image_path) as pil_img:
        phash_obj = imagehash.phash(pil_img)
    # Chuyển chuỗi hex 64-bit sang số nguyên để đếm bit khác nhau bằng phép XOR nhanh
    return md5_str, int(str(phash_obj), 16)


def build_hash_records(image_dir):
    """Duyệt toàn bộ ảnh đã chuẩn hóa và tính mã băm cho từng ảnh.
    Đầu vào: image_dir (Path) - thư mục chứa ảnh processed/all/images.
    Đầu ra: danh sách dict chứa tên file, nguồn, md5 và phash_int.
    """
    image_files = sorted(p for p in image_dir.iterdir() if p.is_file())
    records = []
    for idx, img_path in enumerate(image_files, start=1):
        md5_str, phash_int = compute_hashes(img_path)
        records.append({"file": img_path.name, "source": img_path.name.split("_")[0], "md5": md5_str, "phash": phash_int})
        if idx % PROGRESS_STEP == 0 or idx == len(image_files):
            print(f"[Tính hash] Đã xử lý {idx}/{len(image_files)} ảnh...")
    return records


def find_duplicate_pairs(records):
    """So sánh từng cặp ảnh bằng vòng for đơn giản kết hợp phép toán bit XOR.
    Hai số nguyên 64-bit khi XOR (^) sẽ bật bit 1 ở những vị trí khác nhau;
    đếm số bit 1 chính là khoảng cách Hamming giữa hai mã pHash.
    """
    duplicates, total = [], len(records)
    for i in range(total):
        rec_a = records[i]
        for j in range(i + 1, total):
            rec_b = records[j]
            # Phép XOR tìm các bit khác nhau và bit_count() đếm tổng số bit lệch
            hamming_dist = (rec_a["phash"] ^ rec_b["phash"]).bit_count()
            if rec_a["md5"] == rec_b["md5"]:
                duplicates.append({"file_1": rec_a["file"], "source_1": rec_a["source"], "file_2": rec_b["file"], "source_2": rec_b["source"], "match_type": "exact_md5", "hamming_distance": 0})
            elif hamming_dist <= PHASH_THRESHOLD:
                duplicates.append({"file_1": rec_a["file"], "source_1": rec_a["source"], "file_2": rec_b["file"], "source_2": rec_b["source"], "match_type": "near_phash", "hamming_distance": hamming_dist})
        if (i + 1) % PROGRESS_STEP == 0 or (i + 1) == total:
            print(f"[So sánh cặp] Đã đối chiếu {i + 1}/{total} ảnh...")
    return duplicates


def save_side_by_side_samples(duplicates, image_dir, sample_dir):
    """Ghép cạnh nhau ngẫu nhiên 10 cặp gần trùng để kiểm tra trực quan bằng mắt.
    Đầu vào: danh sách cặp trùng, thư mục ảnh gốc, thư mục xuất ảnh mẫu.
    Đầu ra: số lượng ảnh ghép đã lưu thành công.
    """
    sample_dir.mkdir(parents=True, exist_ok=True)
    near_pairs = [d for d in duplicates if d["match_type"] == "near_phash"] or duplicates
    chosen = random.Random(42).sample(near_pairs, min(SAMPLE_COUNT, len(near_pairs)))
    for idx, pair in enumerate(chosen, start=1):
        p1, p2 = image_dir / pair["file_1"], image_dir / pair["file_2"]
        # Đọc ảnh bằng np.fromfile + imdecode để không lỗi với đường dẫn tiếng Việt
        img1 = cv2.imdecode(np.fromfile(str(p1), dtype=np.uint8), cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(np.fromfile(str(p2), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img1 is not None and img2 is not None:
            combined = np.hstack([cv2.resize(img1, (400, 300)), cv2.resize(img2, (400, 300))])
            out_file = sample_dir / f"pair_{idx:02d}_dist{pair['hamming_distance']}_{pair['file_1']}_vs_{pair['file_2']}"
            cv2.imencode(".jpg", combined)[1].tofile(str(out_file))
    return len(chosen)


def run_duplicate_search():
    """Chạy toàn bộ quy trình phát hiện ảnh trùng và xuất báo cáo thống kê.
    Đầu vào: Không có.
    Đầu ra: danh sách các cặp ảnh trùng/gần trùng.
    """
    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    duplicates = find_duplicate_pairs(build_hash_records(IMAGE_DIR))
    fields = ["file_1", "source_1", "file_2", "source_2", "match_type", "hamming_distance"]
    with open(REPORT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(duplicates)
    saved = save_side_by_side_samples(duplicates, IMAGE_DIR, SAMPLE_DIR)
    cross_source = sum(1 for d in duplicates if d["source_1"] != d["source_2"])
    exact_count = sum(1 for d in duplicates if d["match_type"] == "exact_md5")
    print(f"\n=== KẾT QUẢ KIỂM TRA ẢNH TRÙNG (Ngưỡng pHash <= {PHASH_THRESHOLD}) ===")
    print(f"- Tổng cặp trùng/gần trùng: {len(duplicates)} (MD5: {exact_count}, pHash: {len(duplicates) - exact_count})")
    print(f"- Số cặp trùng GIỮA HAI NGUỒN (kaggle vs roboflow): {cross_source}")
    print(f"- Đã lưu {saved} ảnh ghép mẫu tại: {SAMPLE_DIR}")
    return duplicates


if __name__ == "__main__":
    run_duplicate_search()
