# services/drill_service.py
import os
import sqlite3
from typing import List, Dict
from config import Config


def _get_conn():
    conn = sqlite3.connect(os.path.join(Config.BASE_DIR, Config.DB_FILE_NAME))
    conn.row_factory = sqlite3.Row
    return conn


def get_drill_rows_for_session(
    user_id: str, session_id: str, drill_slug: str
) -> List[Dict]:
    """
    Pull all rows for the given user_id & session_id
    from the table corresponding to drill_slug.
    """
    sql = f"""
        SELECT *
          FROM {drill_slug}
         WHERE user_id = ?
           AND session_id = ?
         ORDER BY timestamp
    """
    with _get_conn() as conn:
        cur = conn.execute(sql, (user_id, session_id))
        return [dict(row) for row in cur.fetchall()]
