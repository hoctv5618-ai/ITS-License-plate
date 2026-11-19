from contextlib import closing
from datetime import datetime
from pathlib import Path
import shutil
import sqlite3
import sys

from config import DB_BACKUP_PATH, DB_PATH, RESULT_DIR

sys.stdout.reconfigure(encoding="utf-8")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, source_type TEXT, file_name TEXT,
    text_raw TEXT, text_clean TEXT, is_valid INTEGER, is_corrected INTEGER DEFAULT 0,
    plate_type TEXT, province_code TEXT, det_conf REAL, ocr_conf REAL,
    image_path TEXT, crop_path TEXT, process_ms REAL
)
"""


def init_db(db_path=DB_PATH):
    """Khởi tạo cơ sở dữ liệu SQLite và tạo bảng detections nếu chưa tồn tại."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(str(db_path))) as conn:
        conn.execute(CREATE_TABLE_SQL)
        conn.commit()
    return True


def save_result(record, db_path=DB_PATH):
    """Lưu một bản ghi nhận diện biển số vào bảng detections bằng câu lệnh có tham số (?) chống SQL Injection."""
    init_db(db_path)
    ts = record.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    clean_txt = str(record.get("text_clean", ""))
    prov = record.get("province_code") or (clean_txt[:2] if record.get("is_valid") and len(clean_txt) >= 2 else "")
    sql = "INSERT INTO detections (timestamp, source_type, file_name, text_raw, text_clean, is_valid, is_corrected, plate_type, province_code, det_conf, ocr_conf, image_path, crop_path, process_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    params = (
        ts, record.get("source_type", "image"), record.get("file_name", ""), record.get("text_raw", ""),
        clean_txt, int(bool(record.get("is_valid", False))), int(bool(record.get("is_corrected", False))),
        record.get("plate_type", ""), prov, float(record.get("det_conf", 0.0)), float(record.get("ocr_conf", 0.0)),
        str(record.get("image_path", "")), str(record.get("crop_path", "")), float(record.get("process_ms", 0.0)),
    )
    with closing(sqlite3.connect(str(db_path))) as conn:
        cur = conn.execute(sql, params)
        conn.commit()
        return cur.lastrowid


def load_history(filters=None, db_path=DB_PATH):
    """Truy vấn lịch sử nhận diện có hỗ trợ lọc theo chuỗi biển số, ngày, nguồn và trạng thái hợp lệ.
    Dùng danh sách tham số (?) động để đảm bảo an toàn tuyệt đối trước lỗi SQL Injection.
    """
    init_db(db_path)
    filters = filters or {}
    clauses, params = ["1=1"], []
    if filters.get("plate_query"):
        clauses.append("(text_clean LIKE ? OR text_raw LIKE ?)")
        q = f"%{filters['plate_query'].strip().upper()}%"
        params.extend([q, q])
    if filters.get("date_str"):
        clauses.append("timestamp LIKE ?")
        params.append(f"{filters['date_str']}%")
    if filters.get("source_type") and filters["source_type"] != "all":
        clauses.append("source_type = ?")
        params.append(filters["source_type"])
    if filters.get("is_valid") is not None and filters["is_valid"] != "all":
        clauses.append("is_valid = ?")
        params.append(int(filters["is_valid"]))
    sql = f"SELECT * FROM detections WHERE {' AND '.join(clauses)} ORDER BY id DESC"
    with closing(sqlite3.connect(str(db_path))) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def delete_record(record_id, db_path=DB_PATH):
    """Xóa một bản ghi lịch sử theo id. Đầu vào: id bản ghi. Đầu ra: True nếu xóa thành công."""
    with closing(sqlite3.connect(str(db_path))) as conn:
        cur = conn.execute("DELETE FROM detections WHERE id = ?", (int(record_id),))
        conn.commit()
        return cur.rowcount > 0


def backup_and_clear_history(db_path=DB_PATH, backup_path=DB_BACKUP_PATH):
    """Sao lưu file cơ sở dữ liệu hiện tại sang backup_path rồi xóa toàn bộ dữ liệu thử nghiệm."""
    init_db(db_path)
    src = Path(db_path)
    dst = Path(backup_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(str(src), str(dst))
    with closing(sqlite3.connect(str(db_path))) as conn:
        cur = conn.execute("DELETE FROM detections")
        conn.commit()
        return str(dst), cur.rowcount


def get_stats(db_path=DB_PATH):
    """Tính thống kê tổng quan: tổng số lượt, tỉ lệ hợp lệ (%), top mã tỉnh và thời gian xử lý trung bình (ms)."""
    rows = load_history(db_path=db_path)
    if not rows:
        return {"total": 0, "valid_pct": 0.0, "top_provinces": [], "avg_ms": 0.0}
    valid_cnt = sum(1 for r in rows if r["is_valid"] == 1)
    prov_counts = {}
    for r in rows:
        p = r.get("province_code", "")
        if p:
            prov_counts[p] = prov_counts.get(p, 0) + 1
    top_prov = sorted(prov_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    avg_ms = sum(float(r["process_ms"]) for r in rows) / len(rows)
    return {"total": len(rows), "valid_pct": round(valid_cnt * 100.0 / len(rows), 2), "top_provinces": top_prov, "avg_ms": round(avg_ms, 2)}


if __name__ == "__main__":
    tmp_db = RESULT_DIR.parent / "tmp_test_history.db"
    tmp_bak = RESULT_DIR.parent / "tmp_test_history_backup.db"
    tmp_db.unlink(missing_ok=True)
    tmp_bak.unlink(missing_ok=True)
    init_db(tmp_db)
    rid1 = save_result({"source_type": "image", "file_name": "a.jpg", "text_raw": "30A12345", "text_clean": "30A-123.45", "is_valid": True, "plate_type": "oto_5so", "det_conf": 0.9, "ocr_conf": 0.85, "process_ms": 42.5}, tmp_db)
    rid2 = save_result({"source_type": "video", "file_name": "v.mp4", "text_raw": "59X112345", "text_clean": "59-X1 123.45", "is_valid": True, "plate_type": "xemay_5so", "det_conf": 0.88, "ocr_conf": 0.80, "process_ms": 55.0}, tmp_db)
    all_rows = load_history(db_path=tmp_db)
    filtered = load_history({"plate_query": "30A", "source_type": "image"}, tmp_db)
    st_info = get_stats(tmp_db)
    del_ok = delete_record(rid1, tmp_db)
    after_rows = load_history(db_path=tmp_db)
    bak_path, cleared_cnt = backup_and_clear_history(tmp_db, tmp_bak)
    bak_rows = load_history(db_path=tmp_bak)
    empty_rows = load_history(db_path=tmp_db)
    tmp_db.unlink(missing_ok=True)
    tmp_bak.unlink(missing_ok=True)
    ok = (
        len(all_rows) == 2
        and len(filtered) == 1
        and st_info["total"] == 2
        and del_ok
        and len(after_rows) == 1
        and cleared_cnt == 1
        and len(bak_rows) == 1
        and len(empty_rows) == 0
    )
    print(f"Kiểm thử database.py [{'PASS' if ok else 'FAIL'}]: thêm=2, lọc={len(filtered)}, xóa={del_ok}, sao_lưu_xóa={cleared_cnt}")
