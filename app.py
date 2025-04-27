from flask import (
    Flask,
    session,
    redirect,
    url_for,
    request,
    render_template,
    jsonify,
    Response,
    send_file,
    abort,
)
import subprocess
from functools import wraps

from live_video_module.salute_feed import (
    generate_frames,
)
from live_video_module.kadamchal_feed import (
    kadamtal_generate_frames,
)
from live_video_module.baju_swing_feed import (
    baju_swing_generate_frames,
)
from live_video_module.tej_chal_feed import (
    tej_chal_generate_frames,
)
from live_video_module.slow_chal_feed import (
    slow_chal_generate_frames,
)
from live_video_module.hill_march_feed import (
    hill_march_generate_frames,
)
from models import init_db
from models.drill import DrillType
from models.user_session import (
    create_user_session,
    end_user_session,
    get_all_active_user_sessions,
    update_drill_type,
)
from services.pdf_generator import generate_pdf_from_table
from services.results_service import (
    get_results,
    kadamchal_results,
    baju_swing_result,
    get_hill_march_result,
    getTejChal_result,
)


app = Flask(__name__)
init_db()

tracking_processes = {}

app.secret_key = "koi_secure_random_secret_key"  # Required for sessions


# Custom decorator to protect admin route
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session or session.get("role") != "admin":
            return redirect(url_for("login"))
        return f(*args, **kwargs)

    return decorated_function


@app.route("/download-pdf/<table_name>")
def download_pdf(table_name):
    try:
        filepath = generate_pdf_from_table(table_name)
        return send_file(filepath, as_attachment=True)
    except ValueError:
        return abort(400, description="Invalid table name")
    except Exception as e:
        return abort(500, description=f"Error: {str(e)}")


# Login route
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        # TODO:
        if username == "admin" and password == "admin123":
            session["user"] = username
            session["role"] = "admin"
            return redirect(url_for("admin_video_panel"))
        else:
            return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/salute")
def salute():
    return render_template("salute_tracking.html")


@app.route("/salute/analysis")
def salute_analysis():
    return render_template("salute_tracking_analysis.html")


@app.route("/kadamchal")
def kadamchal():
    return render_template("kadamchal_tracking.html")


@app.route("/kadamchal/analysis")
def kadamchal_analysis():
    return render_template("kadamchal_tracking_analysis.html")


@app.route("/bajuswing")
def bajuswing():
    return render_template("baju_swing_1_tracking.html")


@app.route("/bajuswing/analysis")
def bajuswing_analysis():
    return render_template("baju_swing_1_tracking_analysis.html")


@app.route("/slowmarch")
def slowmarch():
    return render_template("slow_march_tracking.html")


@app.route("/slowmarch/analysis")
def slowmarch_analysis():
    return render_template("slow_march_tracking_analysis.html")


@app.route("/tejchal")
def tejchal():
    return render_template("tej_chal_tracking.html")


@app.route("/tejchal/analysis")
def tejchal_analysis():
    return render_template("tej_chal_tracking_analysis.html")


@app.route("/hillmarch")
def hillmarch():
    return render_template("hill_march_tracking.html")


@app.route("/admin")
def admin_panel():
    return render_template("admin_panel.html")


@app.route("/result")
def index():
    results = get_results()
    return render_template(
        "result_page.html", results=results, tracking=bool(tracking_processes)
    )


@app.route("/kadamchal_result")
def kadamchal_result():
    results = kadamchal_results()
    return render_template(
        "kadamchal_result_page.html", results=results, tracking=bool(tracking_processes)
    )


@app.route("/baju_swing_result")
def baju_swing():
    results = baju_swing_result()
    return render_template(
        "baju_swing_result_page.html",
        results=results,
        tracking=bool(tracking_processes),
    )


@app.route("/tejchal_result")
def tejchal_result():
    results = getTejChal_result()
    return render_template(
        "tej_chal_result_page.html", results=results, tracking=bool(tracking_processes)
    )


@app.route("/hill_march_result")
def hill_march_result():
    results = get_hill_march_result()
    return render_template(
        "hill_march_result_page.html",
        results=results,
        tracking=bool(tracking_processes),
    )


@app.route("/start_tracking/<mode>")
def start_tracking(mode):
    global tracking_processes

    if mode not in [
        "salute",
        "kadamchal",
        "baju_swing_1",
        "tejchal",
        "slowmarch",
        "hillmarch",
    ]:
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
@app.route("/pose")
def pose():
    return render_template("pose.html")


@app.route("/salue_live_feed")
def salue_live_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/kadamchal_live_feed")
def kadamchal_live_feed():
    return Response(
        kadamtal_generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/bajuswing_live_feed")
def bajuswing_live_feed():
    return Response(
        baju_swing_generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/hill_march_live_feed")
def hill_march_live_feed():
    return Response(
        hill_march_generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/tej_chal_live_feed")
def tej_chal_live_feed():
    return Response(
        tej_chal_generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/slow_chal_live_feed")
def slow_chal_live_feed():
    return Response(
        slow_chal_generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/admin_video_panel")
@admin_required
def admin_video_panel():
    return render_template("admin_video_panel.html")


@app.route("/sessions", methods=["POST"])
def create_session():
    data = request.json
    user_id = data.get("user_id")
    first_name = data.get("first_name")
    last_name = data.get("last_name")
    drill_type = data.get("drill_type")

    if drill_type:
        drill_type = DrillType(drill_type)  # Ensure it matches the Enum

    create_user_session(user_id, first_name, last_name, drill_type)

    return jsonify({"message": "User session created successfully."}), 201


@app.route("/sessions/<user_id>/end", methods=["POST"])
def end_session(user_id):
    end_user_session(user_id)
    return jsonify({"message": "User session ended successfully."}), 200


@app.route("/sessions/active", methods=["GET"])
def get_active_sessions():
    sessions = get_all_active_user_sessions()
    return jsonify([session.dict() for session in sessions]), 200


@app.route("/sessions/<user_id>/drill-type", methods=["PATCH"])
def update_session_drill_type(user_id):
    data = request.json
    drill_type = data.get("drill_type")

    if not drill_type:
        return jsonify({"error": "drill_type is required"}), 400

    update_drill_type(user_id, DrillType(drill_type))
    return jsonify({"message": "Drill type updated successfully."}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=80)
    # serve(app, host="0.0.0.0", port=80)
