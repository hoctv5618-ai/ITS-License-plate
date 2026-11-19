import math
import cv2

from config import FRAME_SKIP, VOTE_MAX_DIST, VOTE_WINDOW
from evaluate_ocr import strip_plate_symbols
from evaluate_utils import compute_iou
from preprocess import to_grayscale
from validate import validate_plate


def check_best_shot(frame_shape, box, crop_img):
    """Lọc khung hình tốt nhất (Smart OCR Trigger):
    Chỉ bật OCR khi hộp biển số không chạm mép khung hình (cách lề >= 6px),
    kích thước đủ lớn (>= 36x16px) và độ nét Laplacian vượt ngưỡng (>= 22.0)
    để tránh đọc biển bị cắt nửa ở rìa ảnh hoặc bị nhòe chuyển động.
    """
    if crop_img is None or crop_img.size == 0:
        return False, 0.0
    fh, fw = frame_shape[:2]
    x1, y1, x2, y2 = box[:4]
    not_touching_border = (x1 > 6) and (y1 > 6) and (x2 < fw - 6) and (y2 < fh - 6)
    is_big_enough = ((x2 - x1) >= 36) and ((y2 - y1) >= 16)
    gray_crop = to_grayscale(crop_img)
    sharpness = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
    is_good = not_touching_border and is_big_enough and (sharpness >= 22.0)
    return is_good, round(sharpness, 2)


def create_new_track(track_id, box, cx, cy, det_conf, frame_idx, frame_img, crop_img):
    """Khởi tạo đối tượng trạng thái ban đầu cho một Track ID mới phát hiện."""
    return {
        "track_id": track_id,
        "bbox": list(box[:4]),
        "best_bbox": list(box[:4]),
        "best_det_conf": round(det_conf, 4),
        "cx": cx,
        "cy": cy,
        "vx": 0.0,
        "vy": 0.0,
        "last_frame": frame_idx,
        "seen_count": 1,
        "valid_reads": 0,
        "ocr_buffer": [],
        "text_raw": "",
        "text_clean": "",
        "is_valid": False,
        "plate_type": "",
        "province_code": "",
        "det_conf": round(det_conf, 4),
        "ocr_conf": 0.0,
        "best_score": 0.0,
        "best_sharpness": 0.0,
        "best_frame": frame_img.copy(),
        "crop_img": crop_img,
    }


def match_or_create_track(tracks, box, frame_idx, frame_img, crop_img):
    """Theo dõi đa đối tượng: ước lượng vị trí bằng vận tốc (vx, vy) và ghép cặp qua IoU/khoảng cách."""
    x1, y1, x2, y2 = box[:4]
    det_c = float(box[4])
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    max_gap = VOTE_WINDOW * max(1, FRAME_SKIP) * 4

    best_tr = None
    best_dist = 1e9
    for tr in tracks:
        dt = frame_idx - tr["last_frame"]
        if dt <= 0 or dt > max_gap:
            continue
        pred_cx = tr["cx"] + tr["vx"] * min(dt, 6)
        pred_cy = tr["cy"] + tr["vy"] * min(dt, 6)
        dist = math.hypot(cx - pred_cx, cy - pred_cy)
        iou = compute_iou(box[:4], tr["bbox"])
        if (iou >= 0.10 or dist <= VOTE_MAX_DIST * 1.8) and dist < best_dist:
            best_tr = tr
            best_dist = dist

    if best_tr is not None:
        dt = max(1, frame_idx - best_tr["last_frame"])
        best_tr["vx"] = 0.6 * best_tr["vx"] + 0.4 * ((cx - best_tr["cx"]) / dt)
        best_tr["vy"] = 0.6 * best_tr["vy"] + 0.4 * ((cy - best_tr["cy"]) / dt)
        best_tr["bbox"] = box[:4]
        best_tr["cx"] = cx
        best_tr["cy"] = cy
        best_tr["last_frame"] = frame_idx
        best_tr["det_conf"] = round(det_c, 4)
        best_tr["seen_count"] += 1
        return best_tr

    new_tr = create_new_track(len(tracks) + 1, box, cx, cy, det_c, frame_idx, frame_img, crop_img)
    tracks.append(new_tr)
    return new_tr


def vote_track_plate(ocr_buffer):
    """Bầu chọn biển số có tổng trọng số cao nhất từ bộ đệm đọc của một Track ID."""
    weight_by_text = {}
    confs_by_text = {}
    for entry in ocr_buffer:
        txt = entry["text_clean"]
        weight_by_text[txt] = weight_by_text.get(txt, 0.0) + entry["weight"]
        confs_by_text.setdefault(txt, []).append(entry["ocr_conf"])

    best_txt = max(
        weight_by_text.keys(),
        key=lambda t: (
            int(validate_plate(t)["is_valid"]),
            round(weight_by_text[t], 4),
            len(confs_by_text[t]),
        ),
    )
    avg_conf = round(sum(confs_by_text[best_txt]) / len(confs_by_text[best_txt]), 4)
    return best_txt, avg_conf


def update_track_ocr(tr, item, frame_img, sharpness):
    """Cập nhật bộ đệm OCR của track_id, bỏ phiếu theo trọng số và lưu khung hình Best-Shot."""
    s_new = strip_plate_symbols(item["text_clean"])
    if len(s_new) < 5:
        return
    valid_bonus = 2.5 if item["is_valid"] else 0.3
    weight = item["det_conf"] * max(0.1, item["ocr_conf"]) * valid_bonus
    tr["ocr_buffer"].append({
        "text_clean": item["text_clean"],
        "text_raw": item["text_raw"],
        "is_valid": item["is_valid"],
        "ocr_conf": item["ocr_conf"],
        "weight": weight,
    })
    if item["is_valid"] and item["ocr_conf"] >= 0.55:
        tr["valid_reads"] += 1

    best_txt, avg_conf = vote_track_plate(tr["ocr_buffer"])
    val = validate_plate(best_txt)
    tr["text_raw"] = item["text_raw"]
    tr["text_clean"] = val["text_clean"]
    tr["is_valid"] = val["is_valid"]
    tr["plate_type"] = val["plate_type"]
    tr["province_code"] = val["province_code"]
    tr["ocr_conf"] = avg_conf

    shot_score = weight * (1.0 + min(sharpness, 200.0) / 200.0)
    if shot_score >= tr["best_score"]:
        tr["best_score"] = shot_score
        tr["best_sharpness"] = sharpness
        tr["best_frame"] = frame_img.copy()
        tr["crop_img"] = item["crop_img"]
        tr["best_bbox"] = list(item["bbox"])
        tr["best_det_conf"] = item["det_conf"]


def draw_tracked_frame(frame, active_tracks, all_tracks, ocr_runs, total_dets):
    """Vẽ thanh HUD thống kê và khung theo dõi gắn mã ID #<track_id> | Biển số lên khung hình video."""
    vis = frame.copy()
    fh, fw = vis.shape[:2]
    valid_ids = len({t["text_clean"] for t in all_tracks if t["is_valid"]})
    saved_pct = int(round((1.0 - ocr_runs / max(1, total_dets)) * 100))

    cv2.rectangle(vis, (0, 0), (fw, 38), (20, 20, 20), -1)
    hud = (
        f"EDGE ALPR TRACKER | Xe hop le: {valid_ids} | "
        f"Track IDs: {len(all_tracks)} | Tiet kiem OCR: {saved_pct}%"
    )
    cv2.putText(vis, hud, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 200), 2)

    for tr in active_tracks:
        x1, y1, x2, y2 = tr["bbox"]
        color = (0, 230, 0) if tr["is_valid"] else (0, 170, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        plate_str = tr["text_clean"] if tr["text_clean"] else "Dang lay net..."
        lock_tag = " [LOCKED]" if tr["valid_reads"] >= 5 else ""
        tag = f"ID #{tr['track_id']} | {plate_str} ({tr['ocr_conf']:.2f}){lock_tag}"
        ty = max(55, y1 - 8)
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 2)
        cv2.rectangle(vis, (x1, ty - th - 6), (min(fw - 2, x1 + tw + 8), ty + 4), color, -1)
        cv2.putText(vis, tag, (x1 + 4, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (10, 10, 10), 2)
    return vis
