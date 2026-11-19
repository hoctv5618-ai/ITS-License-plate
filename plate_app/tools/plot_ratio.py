import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

# Tính đường dẫn từ vị trí script (nằm trong plate_app/tools/)
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "data" / "processed" / "split" / "train"
HIST_PATH = BASE_DIR / "reports" / "ratio_hist.png"


def collect_train_ratios(train_dir):
    """Thu thập tỉ lệ rộng/cao (w/h theo pixel thật) của mọi box trong tập train.
    Đầu vào: train_dir (Path) - thư mục chứa images và labels của tập train.
    Đầu ra: dict {'kaggle': [...], 'roboflow': [...]} chứa danh sách tỉ lệ w/h.
    """
    ratios = {"kaggle": [], "roboflow": []}
    img_dir, lbl_dir = train_dir / "images", train_dir / "labels"
    for img_path in sorted(img_dir.iterdir()):
        if not img_path.is_file():
            continue
        src = "kaggle" if img_path.name.startswith("kaggle_") else "roboflow"
        with Image.open(img_path) as im:
            img_w, img_h = im.size
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        for line in lbl_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 5:
                # Đổi từ tọa độ chuẩn hóa [0,1] sang số pixel thực tế để có tỉ lệ thật
                ratio = (float(parts[3]) * img_w) / max(1.0, float(parts[4]) * img_h)
                if 0.5 <= ratio <= 6.0:
                    ratios[src].append(ratio)
    return ratios


def analyze_peaks_and_valley(ratios_dict):
    """Tìm hai đỉnh (biển vuông, biển dài) và điểm lõm (valley) trên histogram.
    Đầu vào: dict tỉ lệ w/h của hai nguồn.
    Đầu ra: tuple (đỉnh_biển_vuông, đỉnh_biển_dài, ngưỡng_điểm_lõm).
    """
    all_vals = np.array(ratios_dict["kaggle"] + ratios_dict["roboflow"])
    bins = np.linspace(0.6, 5.6, 26)
    counts, edges = np.histogram(all_vals, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # Tìm đỉnh thứ nhất trong vùng biển vuông [1.0, 2.0] và đỉnh thứ hai vùng biển dài [2.6, 4.6]
    mask_sq = (centers >= 1.0) & (centers <= 2.0)
    mask_lg = (centers >= 2.6) & (centers <= 4.6)
    peak_sq = float(centers[mask_sq][np.argmax(counts[mask_sq])])
    peak_lg = float(centers[mask_lg][np.argmax(counts[mask_lg])])
    # Tìm điểm lõm (số lượng thấp nhất) nằm giữa hai đỉnh để làm ngưỡng chia 2 dòng / 1 dòng
    mask_mid = (centers > peak_sq) & (centers < peak_lg)
    valley = float(centers[mask_mid][np.argmin(counts[mask_mid])])
    for c, cnt in zip(centers, counts):
        print(f"Khoảng w/h ~ {c:4.2f}: {cnt:4d} box")
    return round(peak_sq, 2), round(peak_lg, 2), round(valley, 2)


def plot_histogram(ratios_dict, peak_sq, peak_lg, valley, out_path):
    """Vẽ biểu đồ histogram tỉ lệ w/h tách 2 màu theo nguồn và lưu ra file ảnh.
    Đầu vào: dict tỉ lệ w/h, vị trí 2 đỉnh, điểm lõm và đường dẫn lưu ảnh.
    Đầu ra: Không có.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bins = np.linspace(0.6, 5.6, 51)
    plt.figure(figsize=(10, 5), dpi=120)
    plt.hist(ratios_dict["roboflow"], bins=bins, alpha=0.65, label="roboflow", color="#1f77b4")
    plt.hist(ratios_dict["kaggle"], bins=bins, alpha=0.75, label="kaggle", color="#ff7f0e")
    plt.axvline(valley, color="red", linestyle="--", linewidth=2, label=f"Điểm lõm đề xuất: w/h = {valley}")
    plt.title(f"Phân bố tỉ lệ Rộng/Cao (w/h) trên tập Train (Đỉnh 1: {peak_sq}, Đỉnh 2: {peak_lg})")
    plt.xlabel("Tỉ lệ Rộng / Cao (w / h theo pixel thật)")
    plt.ylabel("Số lượng biển số (box)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


if __name__ == "__main__":
    ratios_data = collect_train_ratios(TRAIN_DIR)
    p_sq, p_lg, valley_val = analyze_peaks_and_valley(ratios_data)
    plot_histogram(ratios_data, p_sq, p_lg, valley_val, HIST_PATH)
    print(f"\nĐỉnh biển 2 dòng (vuông): w/h ~ {p_sq}")
    print(f"Đỉnh biển 1 dòng (dài)  : w/h ~ {p_lg}")
    print(f"Điểm lõm giữa hai đỉnh  : w/h = {valley_val}")
    print(f"Đã lưu biểu đồ tại      : {HIST_PATH}")

