# from flask import Flask, jsonify, render_template
# import sqlite3

# app = Flask(__name__)

# def get_results():
#     conn = sqlite3.connect("salute_results.db")
#     cursor = conn.cursor()
#     cursor.execute("SELECT * FROM results ORDER BY id DESC LIMIT 15")
#     data = cursor.fetchall()
#     conn.close()
#     return data

# @app.route("/")
# def index():
#     results = get_results()
#     return render_template("index.html", results=results)

# @app.route("/api/results")
# def api_results():
#     results = get_results()
#     return jsonify(results)

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, jsonify, render_template
import sqlite3
import subprocess
import threading
import os

app = Flask(__name__)
hand_detection_process = None

def get_results():
    conn = sqlite3.connect("salute_results.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM results ORDER BY id DESC LIMIT 50")
    data = cursor.fetchall()
    conn.close()
    return data

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/track")
def track():
    return render_template("tracking.html")
@app.route("/result")
def index():
    results = get_results()
    return render_template("result_page.html", results=results, tracking=hand_detection_process is not None)

@app.route("/start_tracking")
def start_tracking():
    global hand_detection_process
    if hand_detection_process is None:
        hand_detection_process = subprocess.Popen(["python", "hand_detection.py"])
    return jsonify({"status": "Tracking started"})

@app.route("/stop_tracking")
def stop_tracking():
    global hand_detection_process
    if hand_detection_process is not None:
        hand_detection_process.terminate()
        hand_detection_process = None
    return jsonify({"status": "Tracking stopped"})

@app.route("/api/results")
def api_results():
    results = get_results()
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
