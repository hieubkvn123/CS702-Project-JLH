from flask import Flask, render_template, request, jsonify, session, send_from_directory
import json
import uuid
from datetime import datetime, date
import random
import os

# ─────────────────────────────────────────────
#  GPU CONFIGURATION
# ─────────────────────────────────────────────
# Make both GPUs visible to this process.
# Adjust the IDs to match your hardware (e.g. "0,1", "2,3", etc.)
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")

# After setting CUDA_VISIBLE_DEVICES, the physical GPUs are re-indexed
# as cuda:0 and cuda:1 inside this process.
MUSIC_DEVICE = "cuda:0"   # GPU 0 → MusicGen
LLAMA_DEVICE = "cuda:1"   # GPU 1 → Llama-3.1-8B-Instruct

from music_gen_v2 import MusicDiaryGenerator
from llama_assistant import LlamaAssistant

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATED_MUSIC_DIR = os.path.join(BASE_DIR, "generated_music")

# ─────────────────────────────────────────────
#  MODEL INITIALIZATION  (each on its own GPU)
# ─────────────────────────────────────────────

# Music generation model → GPU 0
music_generator = MusicDiaryGenerator(
    model_name="facebook/musicgen-stereo-small",
    output_dir=GENERATED_MUSIC_DIR,
    static_url_prefix="/generated_music",
    device=MUSIC_DEVICE,
)

# Llama-3.1 assistant → GPU 1
LLAMA_MODEL_PATH = os.environ.get(
    "LLAMA_MODEL_PATH",
    "/home/dataset_model/model/Llama-3.1-8B-Instruct",
)
llama_assistant = LlamaAssistant(
    model_path=LLAMA_MODEL_PATH,
    device=LLAMA_DEVICE,
)

app = Flask(__name__)
app.secret_key = "music-diary-secret-key"

# ─────────────────────────────────────────────
#  DATA STORE  (replace with DB for production)
# ─────────────────────────────────────────────
DIARY_ENTRIES = {}   # { "YYYY-MM-DD": { text, emotion, music_url, summary } }
CONVERSATIONS = {}   # { session_id: [ {role, content}, ... ] }

# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("./index.html")


@app.route("/api/entries", methods=["GET"])
def get_entries():
    """Return all diary entries (for calendar colouring)."""
    return jsonify(DIARY_ENTRIES)


@app.route("/api/entry/<date_str>", methods=["GET"])
def get_entry(date_str):
    """Return a single diary entry for a given date."""
    entry = DIARY_ENTRIES.get(date_str)
    if entry:
        return jsonify({"found": True, "entry": entry})
    return jsonify({"found": False})


@app.route("/api/chat/start", methods=["POST"])
def chat_start():
    """Start a new diary conversation session."""
    session_id = str(uuid.uuid4())
    CONVERSATIONS[session_id] = []

    # Generate opening question via Llama
    opening = llama_assistant.guide_conversation(history=[], user_message="")
    CONVERSATIONS[session_id].append({"role": "assistant", "content": opening})

    return jsonify({"session_id": session_id, "message": opening})


@app.route("/api/chat/message", methods=["POST"])
def chat_message():
    """Send a user message; get back the assistant's next guiding question."""
    data = request.get_json()
    session_id = data.get("session_id")
    user_msg = data.get("message", "")

    if session_id not in CONVERSATIONS:
        return jsonify({"error": "Session not found"}), 404

    history = CONVERSATIONS[session_id]
    history.append({"role": "user", "content": user_msg})

    # Generate reply via Llama
    reply = llama_assistant.guide_conversation(history=history[:-1], user_message=user_msg)
    history.append({"role": "assistant", "content": reply})

    # Mark done after ~8 user turns (adjustable)
    user_turn_count = sum(1 for m in history if m["role"] == "user")
    done = user_turn_count >= 8

    return jsonify({"message": reply, "done": done})


@app.route("/generated_music/<path:filename>")
def serve_generated_music(filename):
    return send_from_directory(GENERATED_MUSIC_DIR, filename)


@app.route("/api/generate", methods=["POST"])
def generate_entry():
    """Summarize conversation, extract emotion, generate music."""
    data = request.get_json()
    session_id = data.get("session_id")
    date_str = data.get("date", str(date.today()))

    if session_id not in CONVERSATIONS:
        return jsonify({"error": "Session not found"}), 404

    history = CONVERSATIONS[session_id]

    # Use Llama for summarization and emotion extraction
    summary = llama_assistant.summarize_diary(history)
    emotion = llama_assistant.extract_emotion(history)

    # Generate music clips via MusicGen
    result = music_generator.generate_clips(
        emotion=emotion,
        num_clips=4,
        max_new_tokens=256,
        base_name=f"{emotion}_{uuid.uuid4().hex[:8]}",
    )

    clips = []
    for clip in result["clips"]:
        clips.append({
            "id": clip["id"],
            "label": clip["label"],
            "url": clip["music_url"],
            "prompt": clip["prompt"],
        })

    print(f"[generate] emotion={emotion}, clips={[c['url'] for c in clips]}")

    DIARY_ENTRIES[date_str] = {
        "date": date_str,
        "summary": summary,
        "emotion": emotion,
        "music_url": None,
        "clips": clips,
        "status": "pending_music",
    }

    return jsonify({
        "summary": summary,
        "emotion": emotion,
        "clips": clips,
        "date": date_str,
    })


@app.route("/api/select_music", methods=["POST"])
def select_music():
    """User picks one of the generated clips; finalise the entry."""
    data = request.get_json()
    date_str = data.get("date")
    clip_id = data.get("clip_id")
    clip_url = data.get("clip_url")

    if date_str not in DIARY_ENTRIES:
        return jsonify({"error": "Entry not found"}), 404

    if not clip_url:
        return jsonify({"error": "clip_url is required"}), 400

    DIARY_ENTRIES[date_str]["music_url"] = clip_url
    DIARY_ENTRIES[date_str]["selected_clip_id"] = clip_id
    DIARY_ENTRIES[date_str]["status"] = "complete"
    DIARY_ENTRIES[date_str].pop("clips", None)

    return jsonify({"success": True, "entry": DIARY_ENTRIES[date_str]})


if __name__ == "__main__":
    app.run(debug=True, port=5000)