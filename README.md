# Music Diary — CS702 Team 7

A Flask web UI for the Music Diary system (UbiComp '24).

## Project Structure

```
music_diary/
├── app.py              ← Flask server + placeholder backend functions
└── templates/
    └── index.html      ← Full frontend (single file, no build step)
```

## Setup

```bash
pip install flask
python app.py
# Open http://localhost:5000
```

## Backend Placeholders (app.py)

Your teammates fill in these four functions:

| Function | What it should do |
|---|---|
| `chatgpt_guide_conversation(history, user_message)` | Call ChatGPT Assistant API; return next guiding question |
| `chatgpt_summarize_diary(conversation)` | Summarize conversation into a <200-word first-person diary entry |
| `chatgpt_extract_emotion(conversation)` | Return the dominant emotion keyword (e.g. "joy", "melancholy") |
| `generate_music(emotion)` | Call Audiogen API; return list of up to 4 clip dicts `{id, label, url}` |

## API Routes

| Method | Route | Description |
|---|---|---|
| GET  | `/api/entries` | All diary entries (for calendar) |
| GET  | `/api/entry/<date>` | Single entry for a date |
| POST | `/api/chat/start` | Start a new guided conversation |
| POST | `/api/chat/message` | Send user message, get assistant reply |
| POST | `/api/generate` | Summarize + extract emotion + generate music |
| POST | `/api/select_music` | Finalise entry with chosen clip |

## User Flow

1. Click **New Entry** or a calendar date → guided Q&A starts
2. Answer ~8 questions from the ChatGPT assistant
3. Click **Generate Diary** → diary summary + emotion + 4 music clips appear
4. Pick a music clip → **Save Diary Entry**
5. Entry appears on the calendar (green dot = complete)
