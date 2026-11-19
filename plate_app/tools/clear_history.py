from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from config import DB_BACKUP_PATH, DB_PATH
from database import backup_and_clear_history


def main():
    """Sao lưu data/plates_history.db sang data/plates_history_backup.db rồi xóa toàn bộ dữ liệu thử nghiệm."""
    bak_path, deleted_cnt = backup_and_clear_history(DB_PATH, DB_BACKUP_PATH)
    print(f"Đã sao lưu CSDL sang: {bak_path}")
    print(f"Đã xóa {deleted_cnt} bản ghi thử nghiệm trong: {DB_PATH}")


if __name__ == "__main__":
    main()

