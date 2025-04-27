import sqlite3


def get_results():
    conn = sqlite3.connect("salute_results.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM results ORDER BY id DESC LIMIT 50")
    data = cursor.fetchall()
    conn.close()
    return data


def kadamchal_results():
    conn = sqlite3.connect("salute_results.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM kadamtal_result ORDER BY id DESC LIMIT 50")
    data = cursor.fetchall()
    conn.close()
    return data


def baju_swing_result():
    conn = sqlite3.connect("salute_results.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM baju_swing_result ORDER BY id DESC")
    data = cursor.fetchall()
    conn.close()
    return data


def get_hill_march_result():
    conn = sqlite3.connect("salute_results.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM hill_march_result ORDER BY id DESC")
    data = cursor.fetchall()
    conn.close()
    return data


def getTejChal_result():
    conn = sqlite3.connect("salute_results.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tej_march_result ORDER BY id DESC")
    data = cursor.fetchall()
    conn.close()
    return data
