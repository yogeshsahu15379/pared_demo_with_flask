from flask import Flask, jsonify, render_template, request,Response
import sqlite3
import subprocess
import json
app = Flask(__name__)
tracking_processes = {}  # To track multiple processes
from pose_logic import frame_worker, get_frame, get_pose_data
import threading
import time

# Start the background thread
t = threading.Thread(target=frame_worker)
t.daemon = True
t.start()

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

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/track")
def track():
    return render_template("tracking.html")

@app.route("/salute")
def salute():
    return render_template("salute_tracking.html")

@app.route("/kadamchal")
def kadamchal():
    return render_template("kadamchal_tracking.html")

@app.route("/bajuswing")
def bajuswing():
    return render_template("baju_swing_1_tracking.html")

@app.route("/slowmarch")
def slowmarch():
    return render_template("slow_march_tracking.html")

@app.route("/tejchal")
def tejchal():
    return render_template("tej_chal_tracking.html")

@app.route("/hillmarch")
def hillmarch():
    return render_template("hill_march_tracking.html")

@app.route("/admin")
def admin_panel():
    return render_template("admin_panel.html")


@app.route("/result")
def index():
    results = get_results()
    return render_template("result_page.html", results=results, tracking=bool(tracking_processes))

@app.route("/kadamchal_result")
def kadamchal_result():
    results = kadamchal_results()
    return render_template("kadamchal_result_page.html", results=results, tracking=bool(tracking_processes))

@app.route("/baju_swing_result")
def baju_swing():
    results = baju_swing_result()
    return render_template("baju_swing_result_page.html", results=results, tracking=bool(tracking_processes))

@app.route("/tejchal_result")
def tejchal_result():
    results = getTejChal_result()
    return render_template("tej_chal_result_page.html", results=results, tracking=bool(tracking_processes))

@app.route("/hill_march_result")
def hill_march_result():
    results = get_hill_march_result()
    return render_template("hill_march_result_page.html", results=results, tracking=bool(tracking_processes))

@app.route("/start_tracking/<mode>")
def start_tracking(mode):
    global tracking_processes

    if mode not in ["salute", "kadamchal","baju_swing_1", "tejchal","slowmarch","hillmarch"]:
        return jsonify({"error": "Invalid mode"}), 400

    if mode in tracking_processes:
        return jsonify({"status": f"{mode} tracking already running"})

    script = {
        "salute": "salute_detection.py",
        "kadamchal": "kadamchal_detection.py",
        "baju_swing_1": "baju_swing_detection.py",
        "tejchal": "tej_chal_detection.py",
        "slowmarch": "slow_march_detection.py",
        "hillmarch": "hill_march_detection.py",
    }.get(mode)
    tracking_processes[mode] = subprocess.Popen(["python", script])

    return jsonify({"status": f"{mode} tracking started"})

@app.route("/stop_tracking/<mode>")
def stop_tracking(mode):
    global tracking_processes

    if mode not in tracking_processes:
        return jsonify({"error": f"No tracking running for {mode}"}), 400

    tracking_processes[mode].terminate()
    del tracking_processes[mode]

    return jsonify({"status": f"{mode} tracking stopped"})

@app.route("/api/results")
def api_results():
    results = get_results()
    return jsonify(results)

# Start background thread on app startup

@app.route('/pose')
def pose():
    return render_template('pose.html')


@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            time.sleep(0.05)  # slight delay to avoid overwhelming the browser

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/pose_data')
# def pose_data():
#     data = get_pose_data()
#     return jsonify(data if data else {"angle": 0, "status": "Loading..."})

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=80)
