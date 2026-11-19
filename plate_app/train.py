from collections import Counter
import csv
from datetime import datetime
import os
import shutil
import sys
import time
from pathlib import Path
from PIL import Image

from config import BASE_DIR, DATA_DIR, DATA_YAML_PATH, MODEL_DIR, SPLIT_DIR, TRAIN_RESULTS_MD

sys.stdout.reconfigure(encoding="utf-8")

# Tham số cấu hình huấn luyện đặt ở đầu file (đổi QUICK_TEST = False khi chạy thật)
QUICK_TEST = False
ALLOW_NON_ASCII_PATH = True
MODEL_NAME = "yolov8n.pt"
IMGSZ = 640
BATCH = 16
EPOCHS = 3 if QUICK_TEST else 80
SEED = 42
WORKERS = 2
PATIENCE = 15
RUN_NAME = f"n{IMGSZ}_quick" if QUICK_TEST else f"n{IMGSZ}"


def check_preconditions(target_weight_path):
    """Kiểm tra file trọng số trùng tên, đường dẫn ngoài ASCII và GPU CUDA trước khi train."""
    if target_weight_path.exists():
        print(f"[DỪNG] File {target_weight_path.name} đã tồn tại. Hãy đổi RUN_NAME để không ghi đè.")
        return None
    yaml_ok = DATA_YAML_PATH.exists() and DATA_YAML_PATH.read_text(encoding="utf-8").isascii()
    if not str(BASE_DIR).isascii() or not yaml_ok:
        if ALLOW_NON_ASCII_PATH:
            print("[CẢNH BÁO] Đường dẫn chứa ký tự ngoài ASCII (có dấu tiếng Việt), vẫn tiếp tục chạy vì ALLOW_NON_ASCII_PATH = True.")
        else:
            print("Hãy chuyển dự án sang đường dẫn không dấu (ví dụ D:\\plate_app) rồi chạy lại split_dataset.py để sinh lại data.yaml")
            return None
    try:
        import torch
        if not torch.cuda.is_available():
            print("[CẢNH BÁO & DỪNG] Không phát hiện GPU CUDA. Vui lòng bật GPU trước khi chạy train.")
            return None
        return torch.cuda.get_device_name(0)
    except ImportError:
        print("[LỖI] Chưa cài đặt PyTorch/Ultralytics.")
        return None


def get_dataset_info():
    """Đếm số ảnh train/val/test và tìm kích thước ảnh gốc phổ biến nhất mỗi nguồn ở tập train."""
    counts = {sp: len(list((SPLIT_DIR / sp / "images").glob("*.*"))) for sp in ["train", "val", "test"]}
    top_sizes = {}
    for src in ["kaggle", "roboflow"]:
        c = Counter(Image.open(p).size for p in (SPLIT_DIR / "train" / "images").glob(f"{src}_*.*"))
        w, h = c.most_common(1)[0][0] if c else (0, 0)
        top_sizes[src] = f"{w}x{h}"
    return counts, top_sizes


def read_best_val_epoch(csv_path):
    """Đọc file results.csv của Ultralytics và tìm epoch có mAP50-95 cao nhất trên tập val."""
    if not csv_path.exists():
        return None
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({k.strip(): v.strip() for k, v in r.items()})
    if not rows:
        return None
    best = max(rows, key=lambda r: float(r.get("metrics/mAP50-95(B)", 0.0)))
    return {
        "best_ep": int(float(best["epoch"])), "stopped_ep": len(rows),
        "p": float(best["metrics/precision(B)"]), "r": float(best["metrics/recall(B)"]),
        "map50": float(best["metrics/mAP50(B)"]), "map95": float(best["metrics/mAP50-95(B)"]),
    }


def evaluate_test_splits(model, imgsz):
    """Chạy đánh giá trên toàn tập test và tách riêng theo từng nguồn (kaggle, roboflow)."""
    m_all = model.val(data=str(DATA_YAML_PATH), split="test", imgsz=imgsz, verbose=False)
    res = {"all": (float(m_all.box.mp), float(m_all.box.mr), float(m_all.box.map50), float(m_all.box.map), float(m_all.speed.get("inference", 0.0)))}
    test_imgs = sorted((SPLIT_DIR / "test" / "images").glob("*.*"))
    for src in ["kaggle", "roboflow"]:
        src_files = [p.as_posix() for p in test_imgs if p.name.startswith(f"{src}_")]
        txt_p, yml_p = DATA_DIR / f"test_{src}.txt", DATA_DIR / f"tmp_{src}.yaml"
        txt_p.write_text("\n".join(src_files) + "\n", encoding="utf-8")
        yml_p.write_text(f"path: {DATA_DIR.as_posix()}\ntrain: test_{src}.txt\nval: test_{src}.txt\ntest: test_{src}.txt\nnc: 1\nnames: ['license_plate']\n", encoding="utf-8")
        m_s = model.val(data=str(yml_p), split="test", imgsz=imgsz, verbose=False)
        res[src] = (len(src_files), float(m_s.box.mp), float(m_s.box.mr), float(m_s.box.map50), float(m_s.box.map))
        yml_p.unlink(missing_ok=True)
    return res


def write_results_file(meta, counts, top_sz, val_b, test_r, run_dir, out_pt):
    """Ghi nối tiếp đầy đủ 8 mục kết quả huấn luyện vào reports/train_results.md."""
    rel_run = Path(os.path.relpath(run_dir, TRAIN_RESULTS_MD.parent)).as_posix()
    ap, ar, am50, am95, spd = test_r["all"]
    kn, kp, kr, km50, km95 = test_r["kaggle"]
    rn, rp, rr, rm50, rm95 = test_r["roboflow"]
    md = (
        f"\n## Lần chạy: `{meta['run_name']}` ({meta['timestamp']})\n"
        f"1. **Cấu hình:** Model gốc: `{MODEL_NAME}` | Epochs đặt/dừng: `{EPOCHS}/{val_b['stopped_ep']}` | "
        f"`imgsz={IMGSZ}`, `batch={BATCH}`, `seed={SEED}`, `WORKERS={WORKERS}`, `PATIENCE={PATIENCE}` | "
        f"Thiết bị: `{meta['gpu']}` | Thời gian: `{meta['mins']:.2f}` phút.\n"
        f"   - *Lưu ý: So sánh 640 và 960 chỉ có ý nghĩa khi cùng seed, cùng model, cùng epochs.*\n"
        f"2. **Dữ liệu:** Train `{counts['train']}`, Val `{counts['val']}`, Test `{counts['test']}` | File: `{DATA_YAML_PATH.as_posix()}` | "
        f"Kích thước gốc phổ biến nhất (train): `kaggle={top_sz['kaggle']}`, `roboflow={top_sz['roboflow']}`.\n"
        f"3. **Val tốt nhất (Epoch {val_b['best_ep']}):** Precision `{val_b['p']:.4f}`, Recall `{val_b['r']:.4f}`, mAP50 `{val_b['map50']:.4f}`, mAP50-95 `{val_b['map95']:.4f}`.\n"
        f"4. **Test chung:** Precision `{ap:.4f}`, Recall `{ar:.4f}`, mAP50 `{am50:.4f}`, mAP50-95 `{am95:.4f}`.\n"
        f"5. **Test tách theo nguồn** *(Tập test kaggle chỉ khoảng 50 ảnh nên số liệu dao động mạnh)*:\n\n"
        f"| Nguồn | Số ảnh | Precision | Recall | mAP50 | mAP50-95 |\n|---|---:|---:|---:|---:|---:|\n"
        f"| kaggle | {kn} | {kp:.4f} | {kr:.4f} | {km50:.4f} | {km95:.4f} |\n"
        f"| roboflow | {rn} | {rp:.4f} | {rr:.4f} | {rm50:.4f} | {rm95:.4f} |\n\n"
        f"6. **Tốc độ suy luận trung bình:** `{spd:.2f}` ms/ảnh.\n"
        f"7. **Trọng số & Biểu đồ:** `{out_pt.as_posix()}`\n"
        f"   - ![results]({rel_run}/results.png) ![confusion]({rel_run}/confusion_matrix.png) ![PR]({rel_run}/PR_curve.png)\n"
        f"8. **Nhận xét:** \n---\n"
    )
    with open(TRAIN_RESULTS_MD, "a", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    target_pt = MODEL_DIR / f"best_{RUN_NAME}.pt"
    gpu_name = check_preconditions(target_pt)
    if gpu_name is not None:
        from ultralytics import YOLO
        t0 = time.time()
        try:
            model = YOLO(MODEL_NAME)
            res = model.train(data=str(DATA_YAML_PATH), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, seed=SEED, workers=WORKERS, patience=PATIENCE, project=str(BASE_DIR / "runs" / "detect"), name=RUN_NAME)
            run_dir = Path(res.save_dir)
            best_pt, csv_p = run_dir / "weights" / "best.pt", run_dir / "results.csv"
            if not best_pt.exists() or not csv_p.exists():
                print("[LỖI] Thiếu file best.pt hoặc results.csv sau khi train!")
            else:
                shutil.copy2(best_pt, target_pt)
                val_best = read_best_val_epoch(csv_p)
                counts, top_sz = get_dataset_info()
                test_res = evaluate_test_splits(YOLO(str(target_pt)), IMGSZ)
                meta = {"run_name": RUN_NAME, "gpu": gpu_name, "mins": (time.time() - t0) / 60.0, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")}
                write_results_file(meta, counts, top_sz, val_best, test_res, run_dir, target_pt)
                print(f"Đã lưu trọng số {target_pt} và ghi kết quả vào {TRAIN_RESULTS_MD}")
        except RuntimeError as err:
            if "out of memory" in str(err).lower():
                print("[LỖI CUDA OOM] Tràn bộ nhớ GPU! Hãy giảm BATCH (vd 8 hoặc 4) hoặc giảm IMGSZ.")
            else:
                raise
