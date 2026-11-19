import sys
from config import (
    OCR_ALLOWLIST,
    OCR_CONF,
    OCR_USE_GPU,
    RULE_STRAIGHTEN,
    RULE_TRIM_EDGE,
    RULE_TWO_HALVES,
    TWO_LINE_RATIO,
)
from preprocess import preprocess_plate
from validate import validate_plate

sys.stdout.reconfigure(encoding="utf-8")


def load_reader(use_gpu=OCR_USE_GPU):
    """Khởi tạo bộ đọc EasyOCR (ngôn ngữ tiếng Anh/chữ số Latin) một lần duy nhất."""
    try:
        import easyocr
        return easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
    except Exception as err:
        print(f"[LỖI] Không thể khởi tạo EasyOCR: {err}")
        return None


def read_text(plate_image, reader):
    """Dùng EasyOCR quét các vùng chữ trên ảnh biển số với danh sách ký tự cho phép."""
    if plate_image is None or plate_image.size == 0 or reader is None:
        return []
    raw_results = reader.readtext(plate_image, allowlist=OCR_ALLOWLIST)
    valid_items = []
    for box, text, conf in raw_results:
        clean_str = text.strip().upper()
        if clean_str and float(conf) >= OCR_CONF:
            valid_items.append((box, clean_str, float(conf)))
    return valid_items


def is_same_line(item, line):
    """Kiểm tra hộp chữ item có nằm cùng dòng ngang với các hộp trong line hay không.
    Nếu 2 hộp chồng lấp ngang >25% bề rộng thì bắt buộc thuộc 2 dòng trên-dưới khác nhau.
    """
    ref_cy = sum(x["cy"] for x in line) / len(line)
    ref_bh = sum(x["bh"] for x in line) / len(line)
    if abs(item["cy"] - ref_cy) >= 0.45 * max(item["bh"], ref_bh):
        return False
    for other in line:
        ov_x = min(item["x2"], other["x2"]) - max(item["x1"], other["x1"])
        min_w = max(1.0, min(item["x2"] - item["x1"], other["x2"] - other["x1"]))
        if ov_x > 0.25 * min_w:
            return False
    return True


def group_into_lines(results):
    """Gom các hộp chữ thành từng dòng theo tâm y, chiều cao cạnh hộp và kiểm tra không chồng lấp ngang."""
    if not results:
        return []
    items = []
    for box, text, conf in results:
        xs, ys = [pt[0] for pt in box], [pt[1] for pt in box]
        edge_h = (abs(box[3][1] - box[0][1]) + abs(box[2][1] - box[1][1])) / 2.0 if len(box) == 4 else (max(ys) - min(ys))
        items.append({"x1": min(xs), "x2": max(xs), "cy": (min(ys) + max(ys)) / 2.0, "bh": max(1.0, edge_h), "text": text, "conf": conf})
    items.sort(key=lambda x: x["cy"])
    lines = []
    for item in items:
        matched = next((ln for ln in lines if is_same_line(item, ln)), None)
        if matched is not None:
            matched.append(item)
        else:
            lines.append([item])
    for line in lines:
        line.sort(key=lambda x: x["x1"])
    lines.sort(key=lambda line: sum(x["cy"] for x in line) / len(line))
    return lines


def read_two_halves(work_img, reader):
    """Tách đôi nửa trên (0..55%) và nửa dưới (45..100%) của biển vuông 2 dòng để đọc riêng từng dòng rồi ghép lại.
    Áp dụng nguyên lý tách dòng 2 tầng trước khi đưa vào CRNN để tránh lỗi nối chéo hoặc mất dòng.
    """
    h, _ = work_img.shape[:2]
    top_res = sorted(read_text(work_img[:int(h * 0.55), :], reader), key=lambda x: min(p[0] for p in x[0]))
    bot_res = sorted(read_text(work_img[int(h * 0.45):, :], reader), key=lambda x: min(p[0] for p in x[0]))
    top_txt, bot_txt = "".join(x[1] for x in top_res).strip(), "".join(x[1] for x in bot_res).strip()
    confs = [x[2] for x in top_res + bot_res]
    if not (top_txt and bot_txt):
        return "", 0.0
    return f"{top_txt} {bot_txt}", round(sum(confs) / len(confs), 4)


def read_plate(
    plate_image,
    reader,
    use_preprocess=True,
    straighten=RULE_STRAIGHTEN,
    trim_edge=RULE_TRIM_EDGE,
    two_halves=RULE_TWO_HALVES,
):
    """Đọc biển số: tiền xử lý -> quét chữ -> nếu là biển vuông 2 dòng chưa hợp lệ thì tách đôi 2 nửa đọc lại."""
    if plate_image is None or plate_image.size == 0:
        return "", 0.0, 0
    work_img = (
        preprocess_plate(plate_image, use_threshold=False, straighten=straighten, trim_edge=trim_edge)
        if use_preprocess
        else plate_image
    )
    lines = group_into_lines(read_text(work_img, reader))
    full_text = " ".join("".join(x["text"] for x in ln) for ln in lines).strip()
    confs = [x["conf"] for ln in lines for x in ln]
    avg_conf = round(sum(confs) / len(confs), 4) if confs else 0.0
    h, w = work_img.shape[:2]
    if two_halves and (w / max(1.0, float(h))) < TWO_LINE_RATIO and not validate_plate(full_text)["is_valid"]:
        sp_txt, sp_conf = read_two_halves(work_img, reader)
        if sp_txt and (validate_plate(sp_txt)["is_valid"] or len(sp_txt) > len(full_text)):
            return sp_txt, sp_conf, 2
    return full_text, avg_conf, len(lines)


if __name__ == "__main__":
    sample_ocr = [
        ([[15, 48], [125, 32], [130, 72], [20, 88]], "865.86", 0.92),
        ([[20, 18], [115, 5], [120, 42], [25, 55]], "60-AA", 0.88),
    ]
    grouped = group_into_lines(sample_ocr)
    merged = " ".join("".join(x["text"] for x in line) for line in grouped)
    print(f"Kiểm tra group_into_lines (2 dòng nghiêng): '{merged}' (Số dòng: {len(grouped)})")
