"""Đo tốc độ xử lý video thực tế của pipeline (YOLO + theo dõi + OCR chọn lọc).

Cách dùng:  python tools/bench_video.py <duong_dan_video> [frame_skip]
In ra: số khung đã xử lý, thời gian, FPS xử lý, FPS tương đương trên video gốc,
số lượt phát hiện, số lượt gọi OCR và tỉ lệ OCR tiết kiệm được.
"""
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DETECT_CONF, FRAME_SKIP  # noqa: E402
from detect import load_yolo_model  # noqa: E402
from ocr import load_reader  # noqa: E402
from pipeline import finalize_video_clusters, step_video_frame  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def main():
    video = Path(sys.argv[1])
    skip = int(sys.argv[2]) if len(sys.argv) > 2 else FRAME_SKIP
    models = {"yolo": load_yolo_model(), "reader": load_reader()}
    cap = cv2.VideoCapture(str(video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(3)), int(cap.get(4))
    # Khởi động GPU (warm-up) để không tính thời gian nạp CUDA lần đầu
    ok, first = cap.read()
    if ok:
        step_video_frame(first, 0, models, [], DETECT_CONF, True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    tracks, idx, processed, dets, ocr_runs = [], 0, 0, 0, 0
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % max(1, skip) == 0:
            _, nd, no = step_video_frame(frame, idx, models, tracks, DETECT_CONF, True)
            dets += nd
            ocr_runs += no
            processed += 1
        idx += 1
    elapsed = time.perf_counter() - t0
    cap.release()
    plates = finalize_video_clusters(tracks)

    print(f"Video: {video.name} | {w}x{h} | {src_fps:.1f} fps | {n_total} khung | dài {n_total / src_fps:.1f} s")
    print(f"FRAME_SKIP = {skip} | Khung đã xử lý: {processed} | Tổng thời gian: {elapsed:.2f} s")
    print(f"FPS xử lý (khung được đưa vào mô hình / giây): {processed / elapsed:.2f}")
    print(f"FPS tương đương trên video gốc (tất cả khung đọc / giây): {idx / elapsed:.2f}")
    print(f"Thời gian trung bình mỗi khung xử lý: {elapsed * 1000 / max(1, processed):.1f} ms")
    saved = 100.0 * (1 - ocr_runs / dets) if dets else 0.0
    print(f"Lượt phát hiện: {dets} | Lượt gọi OCR: {ocr_runs} | OCR tiết kiệm: {saved:.1f}%")
    print(f"Số ID theo dõi: {len(tracks)} | Biển duy nhất sau gộp: {len(plates)}")
    for p in plates:
        print("  -", p.get("text_clean"), "| hợp lệ:", p.get("is_valid"), "| số khung:", p.get("seen_count"))


if __name__ == "__main__":
    main()

