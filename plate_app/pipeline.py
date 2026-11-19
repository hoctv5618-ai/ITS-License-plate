from collections import Counter
import sys
import time
import cv2

from config import DETECT_CONF, FRAME_SKIP, PADDING_RATIO
from detect import crop_plate, detect_plates
from evaluate_ocr import levenshtein_distance, strip_plate_symbols
from ocr import read_plate
from preprocess import preprocess_plate
from validate import validate_plate
from video_tracker import (
    check_best_shot,
    draw_tracked_frame,
    match_or_create_track,
    update_track_ocr,
)

sys.stdout.reconfigure(encoding="utf-8")


def recognize_single_box(image, box, reader, det_ms_per_box, use_preprocess=True):
    """Cắt ảnh biển số của 1 khung bao, tiền xử lý, đọc OCR và kiểm tra định dạng."""
    crop_img = crop_plate(image, box[:4], PADDING_RATIO)
    t_pre0 = time.perf_counter()
    if use_preprocess:
        proc_img = preprocess_plate(crop_img, use_threshold=False)
    else:
        proc_img = crop_img
    pre_ms = (time.perf_counter() - t_pre0) * 1000.0

    t_ocr0 = time.perf_counter()
    raw_txt, ocr_conf, _ = read_plate(proc_img, reader, use_preprocess=False)
    ocr_ms = (time.perf_counter() - t_ocr0) * 1000.0
    v = validate_plate(raw_txt)

    return {
        "bbox": box[:4],
        "det_conf": round(float(box[4]), 4),
        "crop_img": crop_img,
        "proc_img": proc_img,
        "text_raw": v["text_raw"],
        "text_clean": v["text_clean"],
        "is_valid": v["is_valid"],
        "plate_type": v["plate_type"],
        "province_code": v["province_code"],
        "ocr_conf": ocr_conf,
        "detect_ms": det_ms_per_box,
        "preprocess_ms": round(pre_ms, 2),
        "ocr_ms": round(ocr_ms, 2),
        "process_ms": round(det_ms_per_box + pre_ms + ocr_ms, 2),
    }


def process_image(image, models, conf_thres=DETECT_CONF, use_preprocess=True):
    """Chạy luồng nhận diện trên 1 ảnh: phát hiện YOLO -> xử lý từng biển số phát hiện được."""
    if image is None or getattr(image, "size", 0) == 0:
        return []
    t_det0 = time.perf_counter()
    boxes = detect_plates(image, models.get("yolo"), conf_thres=conf_thres)
    det_ms = (time.perf_counter() - t_det0) * 1000.0
    if not boxes:
        return []
    det_ms_per_box = round(det_ms / len(boxes), 2)
    reader = models.get("reader")
    results = []
    for box in boxes:
        item = recognize_single_box(image, box, reader, det_ms_per_box, use_preprocess=use_preprocess)
        results.append(item)
    return results


def build_track_representative(tr, fallback_id):
    """Xây dựng bản ghi đại diện từ bộ đệm đọc của một Track ID."""
    buf = tr.get("ocr_buffer") or tr.get("votes") or []
    counts = Counter(v["text_clean"] for v in buf)
    conf_map = {
        t: sum(v["ocr_conf"] for v in buf if v["text_clean"] == t) / counts[t]
        for t in counts
    }
    best_txt = tr.get("text_clean") or max(
        counts.keys(),
        key=lambda t: (int(validate_plate(t)["is_valid"]), counts[t], conf_map[t]),
    )
    val_res = validate_plate(best_txt)
    rep = dict(tr.get("best_item", tr))
    rep.update({
        "track_id": tr.get("track_id", fallback_id),
        "bbox": tr.get("best_bbox", tr.get("bbox")),
        "det_conf": tr.get("best_det_conf", tr.get("det_conf", 0.0)),
        "text_clean": val_res["text_clean"],
        "is_valid": val_res["is_valid"],
        "plate_type": val_res["plate_type"],
        "province_code": val_res["province_code"],
        "ocr_conf": round(conf_map.get(best_txt, tr.get("ocr_conf", 0.0)), 4),
        "seen_count": tr.get("seen_count", len(buf)),
        "ocr_reads": len(buf),
        "sharpness": tr.get("best_sharpness", 0.0),
        "best_frame": tr["best_frame"],
    })
    return rep


def is_same_plate_rep(existing, candidate):
    """Kiểm tra 2 bản ghi đại diện có cùng biển số (hoặc cùng hợp lệ và chỉ lệch <= 1 ký tự)."""
    if existing["text_clean"] == candidate["text_clean"]:
        return True
    if existing["is_valid"] and candidate["is_valid"]:
        s1 = strip_plate_symbols(existing["text_clean"])
        s2 = strip_plate_symbols(candidate["text_clean"])
        return levenshtein_distance(s1, s2) <= 1
    return False


def finalize_video_clusters(tracks):
    """Tổng hợp danh sách xe/biển duy nhất từ các Track ID, gộp các Track trùng biển khi camera lia qua lại."""
    merged = []
    for tr in tracks:
        buf = tr.get("ocr_buffer") or tr.get("votes") or []
        if not buf:
            continue
        rep = build_track_representative(tr, len(merged) + 1)
        dup = next((m for m in merged if is_same_plate_rep(m, rep)), None)
        if dup is not None:
            dup["seen_count"] += rep["seen_count"]
            dup["ocr_reads"] = dup.get("ocr_reads", 1) + rep["ocr_reads"]
            if rep["ocr_conf"] > dup["ocr_conf"]:
                for key in ("text_clean", "ocr_conf", "det_conf", "best_frame", "crop_img", "bbox", "sharpness"):
                    if key in rep:
                        dup[key] = rep[key]
        else:
            merged.append(rep)
    valid_only = [p for p in merged if p["is_valid"]]
    return valid_only if valid_only else [p for p in merged if p["seen_count"] >= 2]


def step_video_frame(frame, frame_idx, models, tracks, conf_thres, use_preprocess):
    """Xử lý 1 khung hình video: Phát hiện YOLO -> Gắn Track ID -> Lọc Best-Shot -> Kích hoạt OCR."""
    boxes = detect_plates(frame, models.get("yolo"), conf_thres=conf_thres)
    active = []
    ocr_triggered = 0
    for box in boxes:
        crop_img = crop_plate(frame, box[:4], PADDING_RATIO)
        tr = match_or_create_track(tracks, box, frame_idx, frame, crop_img)
        active.append(tr)
        is_good, sharp = check_best_shot(frame.shape, box[:4], crop_img)
        need_ocr = (tr["valid_reads"] < 5) and (is_good or not tr["ocr_buffer"])
        if need_ocr:
            ocr_triggered += 1
            raw_txt, ocr_conf, _ = read_plate(crop_img, models.get("reader"), use_preprocess=use_preprocess)
            v = validate_plate(raw_txt)
            item = {
                "bbox": box[:4],
                "det_conf": round(float(box[4]), 4),
                "crop_img": crop_img,
                "text_raw": v["text_raw"],
                "text_clean": v["text_clean"],
                "is_valid": v["is_valid"],
                "plate_type": v["plate_type"],
                "province_code": v["province_code"],
                "ocr_conf": ocr_conf,
            }
            update_track_ocr(tr, item, frame, sharp)
    return active, len(boxes), ocr_triggered


def process_video(
    video_path,
    models,
    frame_skip=FRAME_SKIP,
    progress_callback=None,
    conf_thres=DETECT_CONF,
    use_preprocess=True,
    out_video_path=None,
):
    """Đọc video, theo dõi đối tượng gắn ID # cố định, kích hoạt OCR thông minh và xuất video kết quả."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = max(8.0, fps / max(1, frame_skip))

    writer = None
    if out_video_path:
        fourcc = cv2.VideoWriter_fourcc(*"vp80")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, out_fps, (fw, fh))

    tracks = []
    frame_idx = 0
    total_dets = 0
    ocr_runs = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % max(1, frame_skip) == 0:
            active, n_det, n_ocr = step_video_frame(
                frame, frame_idx, models, tracks, conf_thres, use_preprocess
            )
            total_dets += n_det
            ocr_runs += n_ocr
            if writer is not None and writer.isOpened():
                annotated = draw_tracked_frame(frame, active, tracks, ocr_runs, total_dets)
                writer.write(annotated)
        frame_idx += 1
        if progress_callback is not None:
            progress_callback(min(1.0, frame_idx / total_frames))

    cap.release()
    if writer is not None:
        writer.release()
    return finalize_video_clusters(tracks)


if __name__ == "__main__":
    import numpy as np

    dummy = np.zeros((200, 300, 3), dtype=np.uint8)
    tr_list = []
    crop_dummy = dummy[40:110, 40:140]
    tr = match_or_create_track(tr_list, [40, 40, 140, 110, 0.9], 0, dummy, crop_dummy)
    for txt, val in [("865.86", False), ("60-AA 865.86", True), ("60-AA 865.86", True)]:
        sample_item = {
            "bbox": [40, 40, 140, 110],
            "det_conf": 0.9,
            "ocr_conf": 0.88,
            "crop_img": crop_dummy,
            "text_raw": txt,
            "text_clean": txt,
            "is_valid": val,
        }
        update_track_ocr(tr, sample_item, dummy, 85.0)
    match_or_create_track(tr_list, [180, 120, 280, 190, 0.75], 9, dummy, crop_dummy)
    voted = finalize_video_clusters(tr_list)
    print(f"Kiểm thử Edge Tracker: ID #{voted[0]['track_id']}, kết quả='{voted[0]['text_clean']}', bbox_best={voted[0]['bbox']}")
