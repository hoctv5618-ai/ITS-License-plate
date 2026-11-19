import csv
import hashlib
import random
import shutil
import sys
from pathlib import Path
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

# Cấu hình đường dẫn và tỉ lệ mục tiêu 80/10/10 ở đầu file
BASE_DIR = Path(__file__).resolve().parent
ALL_DIR = BASE_DIR / "data" / "processed" / "all"
SPLIT_DIR = BASE_DIR / "data" / "processed" / "split"
DUP_CSV = BASE_DIR / "reports" / "duplicates.csv"
YAML_PATH = BASE_DIR / "data" / "data.yaml"
RANDOM_SEED = 42
RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}


def find(parent, item):
    """Tìm phần tử đại diện (gốc) của nhóm chứa ảnh item trong cấu trúc Union-Find.
    Đi ngược theo dict 'parent' lên tới gốc và nén đường đi để tra cứu nhanh.
    """
    while parent[item] != item:
        parent[item] = parent[parent[item]]
        item = parent[item]
    return item


def union(parent, item_a, item_b):
    """Hợp nhất hai nhóm chứa item_a và item_b thành một nhóm chung theo tính bắc cầu.
    Đầu vào: dict parent và tên 2 file ảnh trùng/gần trùng. Đầu ra: Không có.
    """
    root_a, root_b = find(parent, item_a), find(parent, item_b)
    if root_a != root_b:
        parent[root_b] = root_a


def build_groups_by_source(all_images, dup_csv):
    """Gom các ảnh trùng bắc cầu thành nhóm và phân loại nhóm theo từng nguồn.
    Đầu vào: danh sách tên file ảnh và đường dẫn file duplicates.csv.
    Đầu ra: dict chứa danh sách các nhóm của 'kaggle' và 'roboflow'.
    """
    parent = {img_name: img_name for img_name in all_images}
    if dup_csv.exists():
        with open(dup_csv, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row["file_1"] in parent and row["file_2"] in parent:
                    union(parent, row["file_1"], row["file_2"])
    clusters = {}
    for img_name in all_images:
        clusters.setdefault(find(parent, img_name), []).append(img_name)
    source_groups = {"kaggle": [], "roboflow": []}
    for group in clusters.values():
        src = "kaggle" if group[0].startswith("kaggle_") else "roboflow"
        source_groups[src].append(sorted(group))
    return source_groups


def split_one_source(groups, rng):
    """Chia các nhóm của một nguồn theo tỉ lệ 80/10/10 bằng cách ưu tiên đặt nhóm lớn trước.
    Đặt nhóm lớn trước giống như xếp đá tảng vào thùng trước rồi mới lấp sỏi nhỏ (nhóm 1-2 ảnh)
    vào sau, giúp các nhóm nhỏ cuối cùng bù khít phần thiếu mà không làm tập nào bị vọt lố tỉ lệ.
    """
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    total = sum(len(g) for g in groups)
    targets = {sp: total * r for sp, r in RATIOS.items()}
    splits = {"train": [], "val": [], "test": []}
    grp_cnt = {"train": 0, "val": 0, "test": 0}
    for group in groups:
        # Chọn tập đang thiếu nhiều ảnh nhất so với mục tiêu (mục_tiêu - số_ảnh_hiện_có)
        best_sp = max(RATIOS, key=lambda sp: targets[sp] - len(splits[sp]))
        splits[best_sp].extend(group)
        grp_cnt[best_sp] += 1
    return splits, grp_cnt


def copy_files_and_write_yaml(split_map):
    """Copy cặp ảnh-nhãn sang thư mục train/val/test và tạo file data.yaml bằng text.
    Đầu vào: dict ánh xạ tên tập ('train', 'val', 'test') tới danh sách tên ảnh.
    """
    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)
    for sp, file_list in split_map.items():
        img_out, lbl_out = SPLIT_DIR / sp / "images", SPLIT_DIR / sp / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for fname in file_list:
            shutil.copy2(ALL_DIR / "images" / fname, img_out / fname)
            shutil.copy2(ALL_DIR / "labels" / f"{Path(fname).stem}.txt", lbl_out / f"{Path(fname).stem}.txt")
    # Dùng đường dẫn dấu '/' (POSIX) để YOLO đọc đúng trên cả Windows
    yaml_text = f"path: {SPLIT_DIR.as_posix()}\ntrain: train/images\nval: val/images\ntest: test/images\nnc: 1\nnames: ['license_plate']\n"
    YAML_PATH.write_text(yaml_text, encoding="utf-8")


def compute_under2_pct(file_list, source_prefix):
    """Tính phần trăm box có tỉ lệ rộng/cao (pixel thật) < 2.0 cho một nguồn trong một tập.
    Đầu vào: danh sách tên ảnh của tập và tiền tố nguồn ('kaggle_' hoặc 'roboflow_').
    """
    total_boxes, under_2 = 0, 0
    for fname in file_list:
        if not fname.startswith(source_prefix):
            continue
        with Image.open(ALL_DIR / "images" / fname) as im:
            img_w, img_h = im.size
        for line in (ALL_DIR / "labels" / f"{Path(fname).stem}.txt").read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 5:
                total_boxes += 1
                if (float(parts[3]) * img_w) / max(1.0, float(parts[4]) * img_h) < 2.0:
                    under_2 += 1
    return round(under_2 * 100.0 / total_boxes, 2) if total_boxes else 0.0


def verify_and_report(split_map, grp_totals, max_grp_len, max_grp_sp):
    """In 3 bảng thống kê (số ảnh + độ lệch %, số nhóm, % box w/h < 2) và kiểm tra rò rỉ.
    Đầu vào: dict ảnh theo tập, dict số nhóm theo tập, kích thước và vị trí nhóm lớn nhất.
    """
    print("\n=== 1. BẢNG SỐ ẢNH & ĐỘ LỆCH SO VỚI MỤC TIÊU (80/10/10) ===")
    for src in ["kaggle", "roboflow"]:
        c = {sp: sum(1 for f in split_map[sp] if f.startswith(f"{src}_")) for sp in RATIOS}
        tot = sum(c.values())
        info = " | ".join(f"{sp}: {c[sp]:4d} ({c[sp]*100/tot - RATIOS[sp]*100:+.2f}%)" for sp in RATIOS)
        print(f"{src:<8} (Tổng {tot:4d}) -> {info}")
    print(f"\n=== 2. SỐ NHÓM THEO TẬP: {grp_totals} | Nhóm lớn nhất ({max_grp_len} ảnh) ở tập: {max_grp_sp} ===")
    print("\n=== 3. PHẦN TRĂM BOX CÓ TỈ LỆ W/H < 2 (BIỂN GẦN VUÔNG) ===")
    for src in ["kaggle", "roboflow"]:
        pcts = " | ".join(f"{sp}: {compute_under2_pct(split_map[sp], src + '_'):5.2f}%" for sp in RATIOS)
        print(f"{src:<8} -> {pcts}")
    md5s = {}
    for sp, fs in split_map.items():
        md5s[sp] = {hashlib.md5((ALL_DIR / "images" / f).read_bytes()).hexdigest() for f in fs}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    leaked = any((set(split_map[a]) & set(split_map[b])) or (md5s[a] & md5s[b]) for a, b in pairs)
    print("\n=== 4. KIỂM TRA RÒ RỈ (TÊN FILE & MD5): " + ("FAIL ===" if leaked else "PASS ==="))


if __name__ == "__main__":
    rng = random.Random(RANDOM_SEED)
    all_imgs = sorted(p.name for p in (ALL_DIR / "images").iterdir() if p.is_file())
    src_groups = build_groups_by_source(all_imgs, DUP_CSV)
    final_splits, grp_totals = {"train": [], "val": [], "test": []}, {"train": 0, "val": 0, "test": 0}
    for src_name in ["kaggle", "roboflow"]:
        part_sp, part_cnt = split_one_source(src_groups[src_name], rng)
        for sp in RATIOS:
            final_splits[sp].extend(part_sp[sp])
            grp_totals[sp] += part_cnt[sp]
    max_g = max(src_groups["kaggle"] + src_groups["roboflow"], key=len)
    max_sp = next(sp for sp in RATIOS if max_g[0] in final_splits[sp])
    copy_files_and_write_yaml(final_splits)
    verify_and_report(final_splits, grp_totals, len(max_g), max_sp)
