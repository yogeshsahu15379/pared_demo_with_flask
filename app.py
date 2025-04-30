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
from models import init_db
from models.drill import (
    DRILL_SLUG_MAP,
    FEED_GENERATORS,
    DrillType,
    DRILL_TYPE_SCRIPT_MAP,
)
from models.user_session import (
    create_user_session,
    get_all_active_user_sessions,
    get_all_users,
    get_session_by_id,
    get_sessions_for_user,
    update_drill_type,
)
from services.drill_service import get_drill_rows_for_session


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
    return render_template(
        "drill_panel.html",
        drill_slug=drill_slug,
        feed_function="live_feed",
        analysis_mode=False,  # Normal control panel
    )


@app.route("/<drill_slug>/<user_id>/analysis")
def drill_analysis(drill_slug, user_id):
    try:
        drill_type = DRILL_SLUG_MAP.get(drill_slug)
        if not drill_type:
            abort(404)

        return render_template(
            "drill_panel.html",
            drill_slug=drill_slug,
            feed_function="live_feed",
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


@app.route("/live_feed/<drill_slug>")
def live_feed(drill_slug):
    generator = FEED_GENERATORS.get(DRILL_SLUG_MAP.get(drill_slug))
    if not generator:
        abort(404)
    return Response(generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/results")
def results():
    drill_slug = request.args.get("drill_slug", type=str)
    if not drill_slug:
        # Optionally redirect to some default or error
        return "Missing drill_slug", 400

    drill_type = DRILL_SLUG_MAP.get(drill_slug)
    if not drill_type:
        return f"Unknown drill_slug '{drill_slug}'", 400

    user_id = request.args.get("user_id", type=str)
    session_id = request.args.get("session_id", type=str)

    # 1) no user_id → list users
    if not user_id:
        users = get_all_users(drill_type=drill_type)
        return render_template(
            "results.html",
            users=users,
            user=None,
            sessions=None,
            results=None,
            drill_slug=drill_slug,
        )

    # 2) user_id but no session_id → list sessions
    user_sessions = get_sessions_for_user(user_id, drill_type=drill_type)
    print("Sessions: ", user_sessions)
    if not session_id:
        return render_template(
            "results.html",
            users=None,
            user={"user_id": user_id},
            sessions=user_sessions,  # ← pass it under the name the template uses
            results=None,
            drill_slug=drill_slug,
        )

    # 3) both → drill rows
    rows = get_drill_rows_for_session(
        user_id=user_id, session_id=session_id, drill_slug=drill_slug
    )
    print(rows)
    session_obj = get_session_by_id(session_id, drill_type)
    print(session_obj)
    return render_template(
        "results.html",
        users=None,
        user={"user_id": user_id},
        sessions=None,
        session=session_obj,  # ← now has .id and .started_at
        results=rows,
        drill_slug=drill_slug,
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
