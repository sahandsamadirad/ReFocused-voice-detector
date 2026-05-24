"""Flask server — serves the debug frontend and exposes voice state + control."""

from flask import Flask, jsonify, request, send_from_directory

from backend.main import get_voice_service

app = Flask(__name__, static_folder="frontend", static_url_path="/static")
voice = get_voice_service()


@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/api/voice/state")
def voice_state():
    return jsonify(voice.snapshot())


@app.route("/api/voice/enroll", methods=["POST"])
def voice_enroll():
    voice.request_enroll()
    return jsonify({"ok": True})


@app.route("/api/voice/enroll", methods=["DELETE"])
def voice_enroll_reset():
    voice.reset_enroll()
    return jsonify({"ok": True})


@app.route("/api/voice/history", methods=["DELETE"])
def voice_history_clear():
    voice.clear_history()
    return jsonify({"ok": True})


@app.route("/api/voice/context", methods=["GET", "POST"])
def voice_context():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        tasks = data.get("tasks", [])
        if isinstance(tasks, str):
            tasks = [t for t in tasks.splitlines() if t.strip()]
        voice.set_context(list(tasks))
    return jsonify({"tasks": voice.get_context()})


if __name__ == "__main__":
    voice.start()
    try:
        app.run(host="127.0.0.1", port=5050, debug=False, use_reloader=False)
    finally:
        voice.stop()
