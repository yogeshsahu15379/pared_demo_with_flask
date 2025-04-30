import os
import subprocess
from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    render_template,
    request,
)

from config import Config
from live_video_module.salute_feed import generate_frames
from models import init_db
from models.drill import DRILL_SLUG_MAP, DrillType, DRILL_TYPE_SCRIPT_MAP
from models.user_session import create_user_session, get_all_active_user_sessions, update_drill_type


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # initialize extensions
    init_db(app)

    return app

app = create_app()

tracking_processes = {}

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/<drill_slug>")
def drill_panel(drill_slug):
    drill_type = DRILL_SLUG_MAP.get(drill_slug)
    if not drill_type:
        abort(404)

    feed_function = f"{drill_slug}_live_feed"
    return render_template(
        "drill_panel.html",
        drill_slug=drill_slug,
        feed_function=feed_function,
        analysis_mode=False,  # Normal control panel
    )


@app.route("/<drill_slug>/<user_id>/analysis")
def drill_analysis(drill_slug, user_id):
    try:
        drill_type = DRILL_SLUG_MAP.get(drill_slug)
        if not drill_type:
            abort(404)

        feed_function = f"{drill_slug}_live_feed"
        return render_template(
            "drill_panel.html",
            drill_slug=drill_slug,
            feed_function=feed_function,
            analysis_mode=True,  # Analysis (feed view) mode
            user_id=user_id,
        )
    except Exception as e:
        print(e)
        abort(500)


@app.route("/start_tracking/<mode>/<user_id>")
def start_tracking(mode, user_id):
    global tracking_processes

    drill_type = DRILL_SLUG_MAP.get(mode)

    if not drill_type:
        return jsonify({"error": "Invalid drill type"}), 400

    user_session = update_drill_type(user_id, drill_type)

    if not user_session:
        return jsonify({"error": "No session found"}), 400

    if mode in tracking_processes:
        return jsonify({"status": f"{mode} tracking already running"})

    script = DRILL_TYPE_SCRIPT_MAP.get(drill_type)

    tracking_processes[mode] = subprocess.Popen(
        [
            "python",
            os.path.join("services", script),
            "--user_id",
            str(user_id),
            "--user_session_id",
            str(user_session.get("id")),
            "--table_name",
            str(mode),
        ]
    )

    return jsonify({"status": f"{mode} tracking started"})


@app.route("/stop_tracking/<mode>/<user_id>")
def stop_tracking(mode, user_id):
    global tracking_processes

    print(mode, user_id)

    update_drill_type(user_id, None)

    if mode not in tracking_processes:
        return jsonify({"error": f"No tracking running for {mode}"}), 400

    tracking_processes[mode].terminate()
    del tracking_processes[mode]

    return jsonify({"status": f"{mode} tracking stopped"})

# STATIC ROUTES
@app.route("/salute_live_feed")
def salute_live_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )

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

@app.route("/sessions/active", methods=["GET"])
def get_active_sessions():
    sessions = get_all_active_user_sessions()
    return jsonify([session.dict() for session in sessions]), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=80)
