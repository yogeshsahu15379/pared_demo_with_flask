from flask import Flask, session, redirect, url_for, request, render_template, jsonify, Response
import sqlite3
import subprocess
import json
app = Flask(__name__)
tracking_processes = {}  # To track multiple processes
# from salute_feed import frame_worker, get_frame, get_pose_data
# from kadamchal_feed import kadamtal_frame_worker, get_kadamtal_frame
import threading
import time
from functools import wraps

from live_video_module.salute_feed import generate_frames  # ✅ Import from the above file
from live_video_module.kadamchal_feed import kadamtal_generate_frames  # ✅ Import from the above file
from live_video_module.baju_swing_feed import baju_swing_generate_frames  # ✅ Import from the above file
from live_video_module.tej_chal_feed import tej_chal_generate_frames  # ✅ Import from the above file
from live_video_module.slow_chal_feed import slow_chal_generate_frames  # ✅ Import from the above file
from live_video_module.hill_march_feed import hill_march_generate_frames  # ✅ Import from the above file

app.secret_key = 'koi_secure_random_secret_key'  # Required for sessions


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

# Custom decorator to protect admin route
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session or session.get('role') != 'admin':
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Dummy check — real app me DB se check karo
        if username == 'admin' and password == 'admin123':
            session['user'] = username
            session['role'] = 'admin'
            return redirect(url_for('admin_video_panel'))
        else:
            return render_template('login.html', error="Invalid credentials")

    return render_template('login.html')


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


@app.route('/salue_live_feed')
def salue_live_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/kadamchal_live_feed')
def kadamchal_live_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/bajuswing_live_feed')
def bajuswing_live_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/hill_march_live_feed')
def hill_march_live_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/tej_chal_live_feed')
def tej_chal_live_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/slow_chal_live_feed')
def slow_chal_live_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/admin_video_panel')
@admin_required
def admin_video_panel():
    return render_template("admin_video_panel.html")

# @app.route('/pose_data')
# def pose_data():
#     data = get_pose_data()
#     return jsonify(data if data else {"angle": 0, "status": "Loading..."})

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=80)
