# LƯU Ý QUAN TRỌNG: Biển số trong bộ này được cắt theo nhãn thật (khung lý tưởng Ground Truth),
# nên kết quả OCR đo trên bộ này là CẬN TRÊN so với khi cắt theo khung do YOLO dự đoán.

import csv
import random
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import OCR_LABELS_CSV, OCR_SET_DIR, PADDING_RATIO, SPLIT_DIR, TWO_LINE_RATIO
from detect import crop_plate
from image_io import read_image, save_image

sys.stdout.reconfigure(encoding="utf-8")

RANDOM_SEED = 42
TARGET_TOTAL = 150
MIN_HEIGHT_PX = 15
FIELDS = ["file", "source", "plate_type", "height_px", "height_bucket", "subset", "text", "readable", "excluded"]


def build_strata_keys():
    """Tạo danh sách 12 khóa phân tầng (2 nguồn x 2 loại biển x 3 nhóm chiều cao ảnh gốc)."""
    return [(s, p, h) for s in ["kaggle", "roboflow"] for p in ["square", "long"] for h in ["<30px", "30-60px", ">60px"]]


def collect_test_candidates(test_dir):
    """Duyệt tập test, lọc các biển có chiều cao gốc >= 15px và gom vào 12 ô phân tầng."""
    buckets = {k: [] for k in build_strata_keys()}
    for img_p in sorted((test_dir / "images").glob("*.*")):
        img = read_image(img_p)
        if img is None:
            continue
        ih, iw = img.shape[:2]
        src = "kaggle" if img_p.name.startswith("kaggle_") else "roboflow"
        for line in (test_dir / "labels" / f"{img_p.stem}.txt").read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5 or (float(parts[4]) * ih) < MIN_HEIGHT_PX:
                continue
            _, cx, cy, bw, bh = [float(x) for x in parts]
            w_px, h_px = bw * iw, bh * ih
            box = [int(round(cx * iw - w_px / 2)), int(round(cy * ih - h_px / 2)), int(round(cx * iw + w_px / 2)), int(round(cy * ih + h_px / 2))]
            ptype = "square" if (w_px / max(1.0, h_px)) < TWO_LINE_RATIO else "long"
            hb = "<30px" if h_px < 30.0 else ("30-60px" if h_px <= 60.0 else ">60px")
            buckets[(src, ptype, hb)].append({"img_p": img_p, "box": box, "source": src, "plate_type": ptype, "height_px": int(round(h_px)), "height_bucket": hb})
    return buckets


def assign_subsets(rows, rng):
    """Chia mỗi ô trong 12 ô phân tầng thành ~2/3 dev và ~1/3 final (ô <= 4 biển cho hết vào dev).
    Giúp quá trình tinh chỉnh tham số chỉ làm trên dev, giữ final độc lập để báo cáo khách quan.
    """
    by_stratum = {k: [] for k in build_strata_keys()}
    for idx, r in enumerate(rows):
        by_stratum[(r["source"], r["plate_type"], r["height_bucket"])].append(idx)
    for k, indices in by_stratum.items():
        pool = list(indices)
        rng.shuffle(pool)
        # Nếu ô có từ 4 biển trở xuống thì đưa hết vào dev để không làm vụn tập kiểm thử
        dev_count = len(pool) if len(pool) <= 4 else int(round(len(pool) * 2.0 / 3.0))
        dev_set = set(pool[:dev_count])
        for idx in indices:
            rows[idx]["subset"] = "dev" if idx in dev_set else "final"
    return rows


def update_existing_or_create(rng):
    """Nếu đã có đủ 150 ảnh trong data/ocr_set/ thì giữ nguyên tên file và chỉ bổ sung cột subset."""
    if OCR_LABELS_CSV.exists() and OCR_SET_DIR.exists() and len(list(OCR_SET_DIR.glob("*.jpg"))) == TARGET_TOTAL:
        with open(OCR_LABELS_CSV, "r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        rows = assign_subsets(rows, rng)
    else:
        buckets = collect_test_candidates(SPLIT_DIR / "test")
        rows = []
        OCR_SET_DIR.mkdir(parents=True, exist_ok=True)
        for k in build_strata_keys():
            rng.shuffle(buckets[k])
            for item in buckets[k][:15]:
                idx = len(rows) + 1
                fname = f"ocr_{idx:03d}_{item['source']}_{item['plate_type']}_{item['height_px']}px.jpg"
                save_image(crop_plate(read_image(item["img_p"]), item["box"], PADDING_RATIO), OCR_SET_DIR / fname)
                rows.append({"file": fname, "source": item["source"], "plate_type": item["plate_type"], "height_px": item["height_px"], "height_bucket": item["height_bucket"], "subset": "dev", "text": "", "readable": ""})
        rows = assign_subsets(rows[:TARGET_TOTAL], rng)
    with open(OCR_LABELS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def print_subset_summary(rows):
    """In bảng đếm số lượng biển theo subset (dev/final) x nguồn x nhóm chiều cao ảnh gốc."""
    print("\n=== BẢNG PHÂN BỐ BỘ NHÃN OCR (SUBSET x NGUỒN x NHÓM CHIỀU CAO) ===")
    for sub in ["dev", "final"]:
        sub_rows = [r for r in rows if r["subset"] == sub]
        print(f"* Tập {sub.upper():<5} (Tổng: {len(sub_rows):3d} biển):")
        for src in ["kaggle", "roboflow"]:
            counts = [f"{hb}: {sum(1 for r in sub_rows if r['source']==src and r['height_bucket']==hb):2d}" for hb in ["<30px", "30-60px", ">60px"]]
            print(f"  - {src:<8} -> " + " | ".join(counts))


if __name__ == "__main__":
    final_rows = update_existing_or_create(random.Random(RANDOM_SEED))
    print_subset_summary(final_rows)
