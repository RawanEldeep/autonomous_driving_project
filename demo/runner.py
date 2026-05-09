"""
Simulation runner for the E4 demo.

Loads AttentionRegularizedPPO (e4_attn_model.zip) and runs the highway-env
simulation in a background thread. Exposes shared state (frame, action,
probabilities, stats) for the Flask server to read.
"""

import base64
import io
import os
import sys
import threading
import time

import numpy as np

# Prevent pygame from opening a display window (we use rgb_array rendering)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# Add autonomous_driving_project root to path so src.* imports work
_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_DEMO_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import gymnasium as gym
import torch
import highway_env  # noqa: F401 — registers highway-v0 with gymnasium

from src.models.baseline_model import CustomExtractor          # must import before load()
from src.models.custom_policy import AttentionRegularizedPPO  # must import before load()

# ---------------------------------------------------------------------------
# Config — hard-coded to match exactly what e4_attn_model.zip was trained on
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join(_PROJECT_ROOT, "ablation_logs", "e4_attn_model.zip")

ACTION_NAMES = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]

_ENV_CONFIG = {
    "lanes_count": 3,
    "vehicles_count": 15,
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "absolute": False,
    },
    "policy_frequency": 2,
    "duration": 200,
    "offscreen_rendering": True,  # render to pygame.Surface directly — no display driver needed
}

# ---------------------------------------------------------------------------
# Shared state — written by simulation thread, read by Flask
# ---------------------------------------------------------------------------

_state: dict = {
    "running": False,
    "frame_b64": None,
    "action": None,
    "action_idx": None,
    "probs": None,
    "step": 0,
    "cumulative_reward": 0.0,
    "episode": 1,
}
_lock = threading.Lock()
_stop_event = threading.Event()
_thread: threading.Thread | None = None

# ---------------------------------------------------------------------------
# Model — loaded once at Flask startup
# ---------------------------------------------------------------------------

_model: AttentionRegularizedPPO | None = None


def load_model() -> None:
    global _model
    if _model is not None:
        return
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    _model = AttentionRegularizedPPO.load(MODEL_PATH)
    _model.policy.set_training_mode(False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _frame_to_b64(frame: np.ndarray) -> str:
    from PIL import Image
    img = Image.fromarray(frame.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _get_probs(obs: np.ndarray) -> list[float]:
    obs_tensor, _ = _model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        dist = _model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.cpu().numpy()
    if probs.ndim > 1:
        probs = probs[0]
    return probs.tolist()


# ---------------------------------------------------------------------------
# Simulation loop (runs in background thread)
# ---------------------------------------------------------------------------

def _get_frame(env) -> np.ndarray:
    """
    highway-env creates the viewer lazily and sets viewer.enabled=False,
    so env.render() returns a black array. Fix: force enabled=True and
    drive the viewer directly via display() + get_image().
    """
    # First call to render() creates the viewer object
    env.render()
    v = env.unwrapped.viewer
    v.enabled = True
    v.display()
    return v.get_image()


def _simulate(stop_event: threading.Event) -> None:
    env = gym.make("highway-v0", render_mode="rgb_array")
    env.unwrapped.configure(_ENV_CONFIG)
    obs, _ = env.reset()

    # Prime the viewer so it exists before the loop
    env.render()

    step = 0
    cumulative_reward = 0.0
    episode = 1

    while not stop_event.is_set():
        probs = _get_probs(obs)
        action, _ = _model.predict(obs, deterministic=True)
        action_idx = int(action)

        obs, reward, done, truncated, _ = env.step(action)
        step += 1
        cumulative_reward += float(reward)

        frame = _get_frame(env)
        frame_b64 = _frame_to_b64(frame)

        with _lock:
            _state["running"] = True
            _state["frame_b64"] = frame_b64
            _state["action"] = ACTION_NAMES[action_idx]
            _state["action_idx"] = action_idx
            _state["probs"] = probs
            _state["step"] = step
            _state["cumulative_reward"] = round(cumulative_reward, 2)
            _state["episode"] = episode

        if done or truncated:
            obs, _ = env.reset()
            episode += 1
            step = 0
            cumulative_reward = 0.0

        time.sleep(0.04)  # ~25 fps

    env.close()
    with _lock:
        _state["running"] = False


# ---------------------------------------------------------------------------
# Public API called by app.py
# ---------------------------------------------------------------------------

def start() -> None:
    global _thread, _stop_event
    if _thread and _thread.is_alive():
        return
    _stop_event = threading.Event()
    _thread = threading.Thread(target=_simulate, args=(_stop_event,), daemon=True)
    _thread.start()


def stop() -> None:
    _stop_event.set()


def get_state() -> dict:
    with _lock:
        return dict(_state)
