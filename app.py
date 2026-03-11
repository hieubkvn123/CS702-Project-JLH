from flask import Flask, render_template, request, jsonify, session
import json
import uuid
from datetime import datetime, date
import random

app = Flask(__name__)
app.secret_key = "music-diary-secret-key"

# ─────────────────────────────────────────────
#  PLACEHOLDER DATA STORE  (replace with DB)
# ─────────────────────────────────────────────
DIARY_ENTRIES = {}   # { "YYYY-MM-DD": { text, emotion, music_url, summary } }
CONVERSATIONS = {}   # { session_id: [ {role, content}, ... ] }

# ─────────────────────────────────────────────
#  PLACEHOLDER BACKEND FUNCTIONS
# ─────────────────────────────────────────────

def chatgpt_guide_conversation(history: list, user_message: str) -> str:
    """
    PLACEHOLDER — Replace with real ChatGPT Assistant API call.

    Should:
    - Take conversation history + user_message
    - Act as a "good listener" asking short guiding questions
    - Conclude within ~15 turns
    - Return assistant reply as a string
    """
    placeholder_questions = [
        "How was your day overall? Tell me something that stood out.",
        "That sounds interesting! How did that make you feel?",
        "Was there a particular moment today that you'd like to remember?",
        "Did you spend time with anyone special today?",
        "What's one thing you're grateful for from today?",
        "Is there anything that's been on your mind that you'd like to reflect on?",
        "How are you feeling right now compared to this morning?",
        "That's a lovely reflection. Is there anything else you'd like to add to your diary?",
    ]
    turn = len([m for m in history if m["role"] == "assistant"])
    if turn >= len(placeholder_questions):
        return "Thank you for sharing! I have enough to create your diary entry now. Click 'Generate Diary' when you're ready."
    return placeholder_questions[turn % len(placeholder_questions)]


def chatgpt_summarize_diary(conversation: list) -> str:
    """
    PLACEHOLDER — Replace with ChatGPT API call that:
    - Reads full conversation
    - Returns a first-person diary entry < 200 words
    """
    return (
        "Today was a meaningful day filled with quiet moments of reflection. "
        "I found myself thinking about the small joys that often go unnoticed — "
        "a warm cup of coffee in the morning, a kind word from a friend, "
        "and the peaceful feeling of the evening settling in. "
        "Emotions ran the full range today, from a touch of uncertainty to a deep sense of gratitude. "
        "There is beauty in the ordinary, and today reminded me to pause and appreciate it."
    )


def chatgpt_extract_emotion(conversation: list) -> str:
    """
    PLACEHOLDER — Replace with ChatGPT API call that:
    - Reads conversation/summary
    - Returns the dominant emotion as a single keyword
      e.g. "joy", "melancholy", "anxiety", "contentment", "excitement"
    """
    emotions = ["joy", "melancholy", "contentment", "excitement", "nostalgia", "calm", "gratitude"]
    return random.choice(emotions)


def generate_music(emotion: str) -> list:
    """
    PLACEHOLDER — Replace with Audiogen (or similar) API call that:
    - Takes an emotion keyword
    - Returns a list of up to 4 music clip URLs (10-sec clips)
    """
    # Return placeholder audio URLs — swap with real generated clips
    return [
        {"id": 1, "label": f"Clip 1 — {emotion.capitalize()} theme", "url": "#"},
        {"id": 2, "label": f"Clip 2 — {emotion.capitalize()} variation", "url": "#"},
        {"id": 3, "label": f"Clip 3 — {emotion.capitalize()} ambient", "url": "#"},
        {"id": 4, "label": f"Clip 4 — {emotion.capitalize()} soft", "url": "#"},
    ]


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
    opening = chatgpt_guide_conversation([], "")
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

    reply = chatgpt_guide_conversation(history, user_msg)
    history.append({"role": "assistant", "content": reply})

    done = len([m for m in history if m["role"] == "user"]) >= 8
    return jsonify({"message": reply, "done": done})


@app.route("/api/generate", methods=["POST"])
def generate_entry():
    """Summarize conversation, extract emotion, generate music."""
    data = request.get_json()
    session_id = data.get("session_id")
    date_str = data.get("date", str(date.today()))

    if session_id not in CONVERSATIONS:
        return jsonify({"error": "Session not found"}), 404

    history = CONVERSATIONS[session_id]

    summary = chatgpt_summarize_diary(history)
    emotion = chatgpt_extract_emotion(history)
    clips = generate_music(emotion)

    # Store (music_url finalised when user picks a clip)
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
    clip_url = data.get("clip_url", "#")

    if date_str not in DIARY_ENTRIES:
        return jsonify({"error": "Entry not found"}), 404

    DIARY_ENTRIES[date_str]["music_url"] = clip_url
    DIARY_ENTRIES[date_str]["selected_clip_id"] = clip_id
    DIARY_ENTRIES[date_str]["status"] = "complete"
    DIARY_ENTRIES[date_str].pop("clips", None)

    return jsonify({"success": True, "entry": DIARY_ENTRIES[date_str]})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
