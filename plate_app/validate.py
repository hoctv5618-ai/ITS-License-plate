# Kiểm tra định dạng dựa trên định dạng biển phổ biến quan sát được, chưa đối chiếu văn bản quy định.
# Hạn chế: biển 4 số có 2 chữ cái, biển quân đội/ngoại giao/công vụ chưa được hỗ trợ.

import re
import sys
from config import (
    MAX_FIXED_CHARS,
    PLATE_PATTERNS,
    RULE_EXTRA_DIGIT_MAP,
    RULE_STRIP_WATERMARK,
    RULE_SWAP_INVERTED_LINES,
    RULE_TRIM_TRAILING_BORDER,
)

sys.stdout.reconfigure(encoding="utf-8")

# Bộ ánh xạ gốc theo thiết kế Giai đoạn 3
BASE_TO_DIGIT = {"O": "0", "D": "0", "I": "1", "Z": "2", "S": "5", "G": "6", "B": "8"}
# Bộ ánh xạ bổ sung từ lỗi thực tế trên clip TikTok (L->4: '60-AK 21L.10', U/Q->0: '6U-B8 935.88', J->3: 'JOK-022.55', T->7)
EXTRA_TO_DIGIT = {"Q": "0", "U": "0", "J": "3", "L": "4", "T": "7"}
TO_DIGIT = {**BASE_TO_DIGIT, **EXTRA_TO_DIGIT}
TO_ALPHA = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B"}


def normalize(
    text,
    strip_watermark=RULE_STRIP_WATERMARK,
    swap_lines=RULE_SWAP_INVERTED_LINES,
    trim_border=RULE_TRIM_TRAILING_BORDER,
):
    """Làm sạch sơ bộ chuỗi OCR với các công tắc bật/tắt từng nhóm luật bổ sung."""
    if not text:
        return ""
    up = str(text).upper().replace("\n", " ").replace("_", "-")
    if strip_watermark:
        # Luật 1 (từ clip TikTok 651f36827ca9c8e82188dd478a650b10.mp4): xóa chữ watermark 'TIKTOK'/'TKTOK'
        up = re.sub(r"T[I1]?KTOK", "", up)
    cleaned = re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9\-\.\s]", "", up)).strip(" -.")
    if trim_border:
        # Luật 3 (từ ảnh Honda City '51G-516.02' đọc thành '616-516.027'): cắt 1 ký tự viền kim loại thừa sau .XX
        cleaned = re.sub(r"\.([A-Z0-9]{2})[17IJT]$", r".\1", cleaned)
    if swap_lines:
        # Luật 2 (từ clip TikTok biển nghiêng '865.8660-AA', '966.6460-B8'): đảo lại khi cụm 5 số đứng trước mã tỉnh
        cleaned = re.sub(r"^([A-Z0-9]{3}\.[A-Z0-9]{2})\s*([A-Z0-9]{2}[\-\.\s]?[A-Z0-9]{2})$", r"\2 \1", cleaned)
    return cleaned


def apply_mask_and_format(
    alnum,
    pattern_name,
    mask,
    max_fixed_chars=MAX_FIXED_CHARS,
    extra_digit_map=RULE_EXTRA_DIGIT_MAP,
):
    """Sửa ký tự nhầm theo vị trí mask ('D'=số, 'L'=chữ) với giới hạn số ký tự sửa tối đa max_fixed_chars."""
    if len(alnum) != len(mask):
        return None
    digit_map = TO_DIGIT if extra_digit_map else BASE_TO_DIGIT
    fixed = []
    subs = 0
    for idx, (ch, req) in enumerate(zip(alnum, mask)):
        orig = ch
        if idx == 3 and pattern_name == "xemay_5so" and alnum[2] == "A":
            ch = {"I": "1", "O": "0"}.get(ch, ch)
        elif req == "D":
            ch = ch if ch.isdigit() else digit_map.get(ch, "")
        else:
            ch = ch if ch.isalpha() else TO_ALPHA.get(ch, "")
        if not ch:
            return None
        if ch != orig:
            subs += 1
        fixed.append(ch)
    s = "".join(fixed)
    if subs > max_fixed_chars or not (11 <= int(s[:2]) <= 99):
        return None
    if pattern_name == "oto_5so":
        return f"{s[:3]}-{s[3:6]}.{s[6:8]}"
    if pattern_name == "oto_4so":
        return f"{s[:3]}-{s[3:7]}"
    if pattern_name in ("xemay_5so", "xemay_5so_2chu"):
        return f"{s[:2]}-{s[2:4]} {s[4:7]}.{s[7:9]}"
    if pattern_name == "xemay_4so":
        return f"{s[:2]}-{s[2:4]} {s[4:8]}"
    return None


def fix_characters(
    text,
    max_fixed_chars=MAX_FIXED_CHARS,
    strip_watermark=RULE_STRIP_WATERMARK,
    swap_lines=RULE_SWAP_INVERTED_LINES,
    trim_border=RULE_TRIM_TRAILING_BORDER,
    extra_digit_map=RULE_EXTRA_DIGIT_MAP,
):
    """Thử sửa ký tự nhầm cho từng mẫu biển số theo cấu hình công tắc luật được truyền vào."""
    norm = normalize(text, strip_watermark=strip_watermark, swap_lines=swap_lines, trim_border=trim_border)
    alnum = re.sub(r"[^A-Z0-9]", "", norm)
    if len(alnum) not in (7, 8, 9):
        return norm, ""
    if swap_lines and re.match(r"^\d{3}\.", norm) and not re.search(r"\.[A-Z0-9]{2}$", norm):
        return norm, ""
    has_dot2 = bool(re.search(r"\.[A-Z0-9]{2}$", norm))
    is_2line_fmt = bool(re.match(r"^[A-Z0-9]{2}\-[A-Z0-9]|^[A-Z0-9]{4}\s", norm))
    is_car_fmt = bool(re.match(r"^[A-Z0-9]{3}\-[A-Z0-9]", norm))
    is_xm4 = len(alnum) == 8 and (not has_dot2) and is_2line_fmt
    for pat in PLATE_PATTERNS:
        pname = pat["name"]
        if (pname in ("oto_4so", "xemay_4so") and has_dot2) or (pname in ("oto_5so", "oto_4so") and is_2line_fmt):
            continue
        if pname.startswith("xemay") and is_car_fmt:
            continue
        if len(alnum) == 8 and ((is_xm4 and pname != "xemay_4so") or (not is_xm4 and pname == "xemay_4so")):
            continue
        candidate = apply_mask_and_format(
            alnum, pname, pat["mask"], max_fixed_chars=max_fixed_chars, extra_digit_map=extra_digit_map
        )
        if candidate and re.match(pat["regex"], candidate):
            out_type = "xemay_5so" if pname == "xemay_5so_2chu" else pname
            return candidate, out_type
    return norm, ""


def validate_plate(
    text,
    max_fixed_chars=MAX_FIXED_CHARS,
    strip_watermark=RULE_STRIP_WATERMARK,
    swap_lines=RULE_SWAP_INVERTED_LINES,
    trim_border=RULE_TRIM_TRAILING_BORDER,
    extra_digit_map=RULE_EXTRA_DIGIT_MAP,
):
    """Kiểm tra tính hợp lệ của biển số, sửa lỗi ký tự nhầm nếu khớp mẫu và trích mã tỉnh."""
    raw_str = "" if text is None else str(text)
    clean_str, ptype = fix_characters(
        raw_str,
        max_fixed_chars=max_fixed_chars,
        strip_watermark=strip_watermark,
        swap_lines=swap_lines,
        trim_border=trim_border,
        extra_digit_map=extra_digit_map,
    )
    is_valid = bool(ptype)
    prov = clean_str[:2] if is_valid else ""
    return {
        "text_raw": raw_str,
        "text_clean": clean_str,
        "is_valid": is_valid,
        "plate_type": ptype,
        "province_code": prov,
    }


if __name__ == "__main__":
    original_cases = [
        ("30A-123.45", True, "30A-123.45", "oto_5so"),
        ("51F-888.88", True, "51F-888.88", "oto_5so"),
        ("3OA-123.4S", True, "30A-123.45", "oto_5so"),
        ("518-123.45", True, "51B-123.45", "oto_5so"),
        ("29A-1234", True, "29A-1234", "oto_4so"),
        ("51N-4G1G", True, "51N-4616", "oto_4so"),
        ("300-9999", True, "30O-9999", "oto_4so"),
        ("59-X1 123.45", True, "59-X1 123.45", "xemay_5so"),
        ("59-XI I23.4S", True, "59-X1 123.45", "xemay_5so"),
        ("43-AA 567.89", True, "43-AA 567.89", "xemay_5so"),
        ("59-B1 0O8.B2", True, "59-B1 008.82", "xemay_5so"),
        ("29-L1 1234", True, "29-L1 1234", "xemay_4so"),
        ("59-XI 1234", True, "59-X1 1234", "xemay_4so"),
        ("60A 359.81", True, "60A-359.81", "oto_5so"),
        ("15A12345", True, "15A-123.45", "oto_5so"),
        ("12345", False, "12345", ""),
        ("30A-12", False, "30A-12", ""),
        ("ABCDEFGH", False, "ABCDEFGH", ""),
        ("00A-123.45", False, "00A-123.45", ""),
        ("", False, "", ""),
        ("59-X1 1234567", False, "59-X1 1234567", ""),
        ("###@@@!!!", False, "", ""),
    ]
    clip_cases = [
        ("51G-516.027", True, "51G-516.02", "oto_5so"),
        ("60-AK 21L.10", True, "60-AK 214.10", "xemay_5so"),
        ("865.8660-AA", True, "60-AA 865.86", "xemay_5so"),
        ("6U-B8 935.88", True, "60-B8 935.88", "xemay_5so"),
        ("93-L1 42.20", False, "93-L1 42.20", ""),
        ("60-M 865.86", False, "60-M 865.86", ""),
        ("93L1 62.20", False, "93L1 62.20", ""),
        ("865.86601", False, "865.86601", ""),
        ("60AK ZIL1", False, "60AK ZIL1", ""),
    ]
    print("=== NHÓM 1: CA KIỂM THỬ GỐC (22 ca) ===")
    orig_pass = 0
    for idx, (inp, ok_e, cl_e, tp_e) in enumerate(original_cases, 1):
        res = validate_plate(inp)
        ok = (res["is_valid"], res["text_clean"], res["plate_type"]) == (ok_e, cl_e, tp_e)
        orig_pass += int(ok)
        print(f"Gốc {idx:02d} [{'PASS' if ok else 'FAIL'}]: '{inp}' -> clean='{res['text_clean']}', valid={res['is_valid']}, type='{res['plate_type']}'")
    print(f"-> Kết quả Nhóm 1 (Gốc): {orig_pass}/{len(original_cases)} ca PASS\n")

    print("=== NHÓM 2: CA MỚI TỪ CLIP & ẢNH THỰC TẾ (9 ca) ===")
    clip_pass = 0
    for idx, (inp, ok_e, cl_e, tp_e) in enumerate(clip_cases, 1):
        res = validate_plate(inp)
        ok = (res["is_valid"], res["text_clean"], res["plate_type"]) == (ok_e, cl_e, tp_e)
        clip_pass += int(ok)
        print(f"Clip {idx:02d} [{'PASS' if ok else 'FAIL'}]: '{inp}' -> clean='{res['text_clean']}', valid={res['is_valid']}, type='{res['plate_type']}'")
    print(f"-> Kết quả Nhóm 2 (Clip): {clip_pass}/{len(clip_cases)} ca PASS")
    total_pass = orig_pass + clip_pass
    total_cnt = len(original_cases) + len(clip_cases)
    print(f"=> TỔNG KẾT validate.py: {total_pass}/{total_cnt} ca PASS (Gốc: {orig_pass}/{len(original_cases)}, Clip: {clip_pass}/{len(clip_cases)})")
