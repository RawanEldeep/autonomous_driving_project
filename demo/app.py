"""
Flask server for the E4 highway demo.

Endpoints:
    GET  /        — serves the demo UI
    POST /start   — starts the simulation thread
    POST /stop    — stops the simulation thread
    GET  /state   — returns current simulation state as JSON
"""

import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", category=UserWarning)

from flask import Flask, jsonify, render_template
import runner

app = Flask(__name__)

# Load the model once at startup (takes ~2 seconds)
print("Loading E4 model...")
runner.load_model()
print("Model ready.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start():
    runner.start()
    return jsonify({"ok": True})


@app.route("/stop", methods=["POST"])
def stop():
    runner.stop()
    return jsonify({"ok": True})


@app.route("/state")
def state():
    return jsonify(runner.get_state())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
