# services/drill_service.py
import os
import sqlite3
from typing import List, Dict
from config import Config
from models.drill import DRILL_SLUG_MAP
from models.user_session import get_session_by_id


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

def find_all_sessions_with_results(user_id: str):
    sessions_with_result = []

    for drill_slug in DRILL_SLUG_MAP.keys():
        try:
            sql = f"""
                SELECT DISTINCT session_id
                FROM {drill_slug}
                WHERE user_id = {user_id}
                ORDER BY timestamp
            """
            with _get_conn() as conn:
                cur = conn.execute(sql)
                session_ids = [row[0] for row in cur.fetchall()]
                for session_id in session_ids:
                    session = get_session_by_id(session_id)
                    # Marking Session result with drill Type
                    session.drill_type = drill_slug
                    sessions_with_result.append(session)
        except sqlite3.OperationalError:
            # Handle the case where the table doesn't exist
            pass
    return sessions_with_result