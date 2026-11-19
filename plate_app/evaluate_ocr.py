import csv
import re
import sys
import time
from config import (
    OCR_ABLATION_CSV,
    OCR_ABLATION_PREPROCESS_CSV,
    OCR_COMPARISON_CSV,
    OCR_LABELS_CSV,
    OCR_RAW_CACHE_CSV,
    OCR_SET_DIR,
    PLATE_PATTERNS,
    RULE_STRAIGHTEN,
    RULE_TRIM_EDGE,
    RULE_TWO_HALVES,
    TWO_LINE_RATIO,
)
from image_io import read_image
from ocr import group_into_lines, load_reader, read_text, read_two_halves
from preprocess import preprocess_plate
from validate import normalize, validate_plate

sys.stdout.reconfigure(encoding="utf-8")


def strip_plate_symbols(text):
    """Chuẩn hóa chuỗi biển số về chỉ gồm chữ cái in hoa và chữ số để so sánh công bằng."""
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def levenshtein_distance(s1, s2):
    """Tính khoảng cách chỉnh sửa Levenshtein (số thao tác thêm, xóa, thay thế ít nhất để biến s1 thành s2)."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if not s2:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            ins = prev_row[j + 1] + 1
            dele = curr_row[j] + 1
            sub = prev_row[j] + (0 if c1 == c2 else 1)
            curr_row.append(min(ins, dele, sub))
        prev_row = curr_row
    return prev_row[-1]


def load_labeled_subset(target_subset):
    """Đọc các dòng đã gán nhãn (readable == '1', có text, và excluded != '1') thuộc tập dev hoặc final."""
    if not OCR_LABELS_CSV.exists():
        return []
    rows = []
    with open(OCR_LABELS_CSV, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if str(r.get("excluded", "0")).strip() == "1":
                continue
            if str(r.get("readable", "")).strip() == "1" and str(r.get("text", "")).strip():
                if r.get("subset", "dev") == target_subset:
                    rows.append(r)
    return rows


def run_one_mode(
    rows,
    reader,
    prep_mode,
    straighten=RULE_STRAIGHTEN,
    trim_edge=RULE_TRIM_EDGE,
    two_halves=RULE_TWO_HALVES,
):
    """Chạy EasyOCR ĐÚNG MỘT LẦN trên tập ảnh cho 1 cấu hình tiền xử lý và trả về kết quả chữ thô."""
    raw_items = []
    for r in rows:
        img = read_image(OCR_SET_DIR / r["file"])
        if img is None:
            continue
        t0 = time.perf_counter()
        if prep_mode == "raw":
            work = img
        else:
            work = preprocess_plate(
                img,
                use_threshold=(prep_mode == "threshold"),
                straighten=straighten,
                trim_edge=trim_edge,
            )
        lines = group_into_lines(read_text(work, reader))
        raw_txt = " ".join("".join(x["text"] for x in ln) for ln in lines).strip()
        h, w = work.shape[:2]
        if prep_mode != "raw" and two_halves and (w / max(1.0, float(h))) < TWO_LINE_RATIO:
            if not validate_plate(raw_txt)["is_valid"]:
                sp_txt, _ = read_two_halves(work, reader)
                if sp_txt and (validate_plate(sp_txt)["is_valid"] or len(sp_txt) > len(raw_txt)):
                    raw_txt = sp_txt
        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        raw_items.append({
            "file": r["file"],
            "subset": r.get("subset", "dev"),
            "source": r["source"],
            "hb": r["height_bucket"],
            "preprocess": prep_mode,
            "raw_txt": raw_txt,
            "gt_text": r["text"],
            "gt_s": strip_plate_symbols(r["text"]),
            "ms": ms,
        })
    return raw_items


def summarize_records(subset_name, prep_mode, use_val, raw_items):
    """Đánh giá use_val (False/True) từ danh sách chữ OCR thô đã tính sẵn mà không chạy lại EasyOCR."""
    recs = []
    for item in raw_items:
        pred_txt = validate_plate(item["raw_txt"])["text_clean"] if use_val else item["raw_txt"]
        gt_s = item["gt_s"]
        pr_s = strip_plate_symbols(pred_txt)
        recs.append({
            "source": item["source"],
            "hb": item["hb"],
            "exact": int(gt_s == pr_s),
            "dist": levenshtein_distance(gt_s, pr_s),
            "gt_len": max(1, len(gt_s)),
            "ms": float(item["ms"]),
        })

    def calc_metrics(sub):
        if not sub:
            return 0.0, 0.0
        ex = round(sum(x["exact"] for x in sub) * 100.0 / len(sub), 2)
        cer = round(sum(x["dist"] for x in sub) * 100.0 / sum(x["gt_len"] for x in sub), 2)
        return ex, cer

    ex_all, cer_all = calc_metrics(recs)
    ex_kag, _ = calc_metrics([x for x in recs if x["source"] == "kaggle"])
    ex_rob, _ = calc_metrics([x for x in recs if x["source"] == "roboflow"])
    ex_u30, _ = calc_metrics([x for x in recs if x["hb"] == "<30px"])
    ex_30_60, _ = calc_metrics([x for x in recs if x["hb"] == "30-60px"])
    ex_o60, _ = calc_metrics([x for x in recs if x["hb"] == ">60px"])
    avg_ms = round(sum(x["ms"] for x in recs) / len(recs), 2) if recs else 0.0
    return {
        "subset": subset_name,
        "preprocess": prep_mode,
        "use_validate": use_val,
        "count": len(recs),
        "exact_pct": ex_all,
        "cer_pct": cer_all,
        "exact_kaggle": ex_kag,
        "exact_roboflow": ex_rob,
        "exact_u30": ex_u30,
        "exact_30_60": ex_30_60,
        "exact_o60": ex_o60,
        "avg_ms": avg_ms,
    }


def eval_ablation_config(config_name, raw_predictions, fn_validate):
    """Đánh giá 1 cấu hình trên danh sách (raw_txt, gt_s) và tính tỉ lệ hợp lệ nhưng đọc SAI."""
    total = len(raw_predictions)
    if total == 0:
        return {}
    exact_cnt = 0
    total_dist = 0
    total_gt_len = 0
    valid_cnt = 0
    false_valid_cnt = 0
    for raw_txt, gt_s in raw_predictions:
        res = fn_validate(raw_txt)
        pr_s = strip_plate_symbols(res["text_clean"])
        is_exact = (pr_s == gt_s)
        is_val = bool(res["is_valid"])
        exact_cnt += int(is_exact)
        total_dist += levenshtein_distance(gt_s, pr_s)
        total_gt_len += max(1, len(gt_s))
        valid_cnt += int(is_val)
        if is_val and not is_exact:
            false_valid_cnt += 1
    return {
        "config": config_name,
        "count": total,
        "exact_match_pct": round(exact_cnt * 100.0 / total, 2),
        "cer_pct": round(total_dist * 100.0 / max(1, total_gt_len), 2),
        "valid_marked_pct": round(valid_cnt * 100.0 / total, 2),
        "false_valid_pct": round(false_valid_cnt * 100.0 / total, 2),
        "false_valid_among_valid_pct": round(false_valid_cnt * 100.0 / valid_cnt, 2) if valid_cnt else 0.0,
    }


def run_ablation_on_dev(raw_predictions):
    """So sánh 7 cấu hình validate trên chữ OCR thô đã cache (chỉ chạy lại validate, không chạy lại EasyOCR)."""
    def no_validate_fn(txt):
        norm = normalize(txt, strip_watermark=False, swap_lines=False, trim_border=False)
        matched = any(re.match(p["regex"], norm) for p in PLATE_PATTERNS)
        return {"text_clean": norm, "is_valid": matched}

    configs = [
        ("a_khong_validate", no_validate_fn),
        ("b_validate_max_fixed_1", lambda t: validate_plate(t, max_fixed_chars=1)),
        ("c_validate_hien_tai", lambda t: validate_plate(t)),
        ("d1_tat_watermark_tiktok", lambda t: validate_plate(t, strip_watermark=False)),
        ("d2_tat_dao_dong", lambda t: validate_plate(t, swap_lines=False)),
        ("d3_tat_cat_vien_sau_xx", lambda t: validate_plate(t, trim_border=False)),
        ("d4_tat_anh_xa_q_u_j_l_t", lambda t: validate_plate(t, extra_digit_map=False)),
    ]
    ablation_rows = [eval_ablation_config(name, raw_predictions, fn) for name, fn in configs]
    with open(OCR_ABLATION_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(ablation_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ablation_rows)
    print(f"Đã xuất bảng khảo sát cắt bỏ luật validate tại: {OCR_ABLATION_CSV}")
    return ablation_rows


def so_sanh_tien_xu_ly(target_subset="dev", ablate_preprocess=False):
    """Thực nghiệm OCR: tính chữ OCR thô 1 lần lưu vào reports/ocr_raw_cache.csv, hoặc chạy --ablate-preprocess."""
    if target_subset == "final":
        print("[CẢNH BÁO] Tập final chỉ được chạy ĐÚNG MỘT LẦN ở cuối cùng để lấy số liệu báo cáo!")
    rows = load_labeled_subset("dev" if ablate_preprocess else target_subset)
    if not rows:
        print(f"[THÔNG BÁO] Chưa có dòng nào có readable=1 (không bị excluded) trong tập '{target_subset}'. Hãy gán nhãn trước.")
        return []

    if ablate_preprocess:
        reader = load_reader()
        prep_configs = [
            ("p0_hien_tai_bat_het", True, True, True),
            ("p1_tat_straighten_hough", False, True, True),
            ("p2_tat_trim_edge_4_5pct", True, False, True),
            ("p3_tat_two_halves", True, True, False),
            ("p4_tat_ca_3_cong_tac", False, False, False),
        ]
        prep_ablation_rows = []
        for cfg_name, st_flag, tr_flag, th_flag in prep_configs:
            items = run_one_mode(rows, reader, "clahe", straighten=st_flag, trim_edge=tr_flag, two_halves=th_flag)
            preds = [(x["raw_txt"], x["gt_s"]) for x in items]
            prep_ablation_rows.append(eval_ablation_config(cfg_name, preds, validate_plate))
        with open(OCR_ABLATION_PREPROCESS_CSV, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(prep_ablation_rows[0].keys()))
            writer.writeheader()
            writer.writerows(prep_ablation_rows)
        print(f"Đã xuất bảng khảo sát tiền xử lý tại: {OCR_ABLATION_PREPROCESS_CSV}")
        return prep_ablation_rows

    cache_map = {}
    if OCR_RAW_CACHE_CSV.exists():
        with open(OCR_RAW_CACHE_CSV, "r", encoding="utf-8-sig") as f:
            for cr in csv.DictReader(f):
                cache_map[(cr["file"], cr["preprocess"])] = cr

    reader = None
    cache_updated = False
    mode_items = {}
    for prep_mode in ["raw", "clahe", "threshold"]:
        missing_rows = [r for r in rows if (r["file"], prep_mode) not in cache_map]
        if missing_rows:
            if reader is None:
                reader = load_reader()
            for item in run_one_mode(missing_rows, reader, prep_mode):
                cache_map[(item["file"], prep_mode)] = item
                cache_updated = True
        mode_items[prep_mode] = [cache_map[(r["file"], prep_mode)] for r in rows if (r["file"], prep_mode) in cache_map]

    if cache_updated or not OCR_RAW_CACHE_CSV.exists():
        all_cached = list(cache_map.values())
        if all_cached:
            cache_fields = ["file", "subset", "source", "hb", "preprocess", "raw_txt", "gt_text", "gt_s", "ms"]
            with open(OCR_RAW_CACHE_CSV, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=cache_fields)
                writer.writeheader()
                for c_item in all_cached:
                    writer.writerow({k: c_item.get(k, "") for k in cache_fields})
            print(f"Đã lưu cache chữ OCR thô tại: {OCR_RAW_CACHE_CSV}")

    summary_rows = []
    for prep_mode in ["raw", "clahe", "threshold"]:
        for use_val in [False, True]:
            summary_rows.append(summarize_records(target_subset, prep_mode, use_val, mode_items[prep_mode]))
    with open(OCR_COMPARISON_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Đã xuất bảng so sánh OCR ({target_subset}) tại: {OCR_COMPARISON_CSV}")
    if target_subset == "dev" and mode_items.get("clahe"):
        clahe_raw_preds = [(x["raw_txt"], x["gt_s"]) for x in mode_items["clahe"]]
        run_ablation_on_dev(clahe_raw_preds)
    return summary_rows


if __name__ == "__main__":
    if "--ablate-preprocess" in sys.argv:
        so_sanh_tien_xu_ly(target_subset="dev", ablate_preprocess=True)
    else:
        sub = "final" if "--final" in sys.argv else "dev"
        so_sanh_tien_xu_ly(target_subset=sub, ablate_preprocess=False)
