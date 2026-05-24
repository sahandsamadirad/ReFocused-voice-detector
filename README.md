# ReFocused — Voice Signal

Mic-based attention signal for ReFocused. Mirrors the gaze module's split:

```
backend/distraction/voice/
├── detector.py    # pure: frames → VAD prob / embeddings / transcript / verdict
└── service.py     # mic loop in a thread, holds latest state behind a lock
```

## Pipeline (all local, no cloud)

```
mic (16 kHz mono)
   ↓ 512-sample frames (32 ms)
Silero VAD                       → speech / silence
   ↓ utterance ≥ MIN_UTTERANCE_MS
Resemblyzer (256-D embedding)    → cosine vs enrolled voice → is_user?
   ↓ if user
faster-whisper (base.en, int8)   → transcript
   ↓
all-MiniLM-L6-v2 (384-D)         → max cosine vs task contexts
   ↓
classify                         → ON_TASK / OFF_TASK / NOT_USER / TOO_SHORT / NO_CONTEXT
```

The classifier is rule-based; Claude is never involved here. Voice is a signal
the rest of ReFocused consumes via `/api/voice/state`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py
```

Open <http://127.0.0.1:5050>.

**First run downloads ~250 MB of models** (Silero, Resemblyzer, faster-whisper
base.en, MiniLM). They're cached in `~/.cache/`.

**macOS mic permission:** the OS will prompt your terminal (Terminal/iTerm/VS
Code) the first time the stream opens. If you don't see the prompt and the UI
shows "mic unavailable", grant access in *System Settings → Privacy & Security
→ Microphone* and restart.

## Using the debug UI

1. Click **Enroll my voice** and talk normally for ~6 seconds.
2. Paste your current task(s) into the textarea, one per line. Click **Update tasks**.
3. Talk. After you stop, the "Last utterance" card shows:
   - the transcript,
   - speaker cosine (vs your enrolled voice),
   - relevance cosine (max vs your task contexts),
   - the verdict.

Without enrollment, every speaker counts as "you". Without tasks, the verdict
is `NO_CONTEXT` (we can't decide on/off-task).

## Verdicts

| Verdict | Meaning |
|---|---|
| `ON_TASK` | You spoke ≥ MIN_UTTERANCE_MS, voice matched, transcript was semantically close to a task. **Not a distraction.** |
| `OFF_TASK` | You spoke, voice matched, but the transcript wasn't related to any task. **Distraction signal.** |
| `NOT_USER` | Someone else was talking. Ignore. |
| `TOO_SHORT` | Brief sound (cough, "huh", "ok") — below MIN_UTTERANCE_MS. Ignore. |
| `NO_CONTEXT` | No task context set yet, so on/off-task is undecidable. |

## Tuning (env vars)

| Var | Default | What |
|---|---|---|
| `VOICE_MIN_UTTERANCE_MS` | `1500` | Below this, an utterance is `TOO_SHORT`. |
| `VOICE_TAIL_SILENCE_MS` | `700` | Silence required to close an utterance. |
| `VOICE_MAX_UTTERANCE_MS` | `20000` | Safety cap on a single utterance. |
| `VOICE_VAD_ON` | `0.55` | Silero prob to start speech. |
| `VOICE_VAD_OFF` | `0.35` | Silero prob below which silence counts. |
| `VOICE_SPEAKER_THRESHOLD` | `0.75` | Cosine ≥ this ⇒ enrolled user. |
| `VOICE_RELEVANCE_THRESHOLD` | `0.40` | Transcript cosine ≥ this ⇒ on-task. |
| `VOICE_WHISPER_MODEL` | `base.en` | Try `tiny.en` for speed, `small.en` for quality. |
| `VOICE_ENROLL_SECONDS` | `6` | Enrollment recording length. |
| `MIC_INDEX` | (default) | Override sounddevice input index. |

Find your mic index with:

```python
python -c "import sounddevice as sd; print(sd.query_devices())"
```

## API

| Endpoint | Method | Body | Returns |
|---|---|---|---|
| `/api/voice/state` | GET | — | Snapshot: `speaking`, `current_utterance_ms`, `enrolled`, `last`, `thresholds`, … |
| `/api/voice/enroll` | POST | — | Start a 6-sec enrollment recording. |
| `/api/voice/enroll` | DELETE | — | Clear the enrolled voice. |
| `/api/voice/context` | GET | — | `{"tasks": [...]}` |
| `/api/voice/context` | POST | `{"tasks": [...]}` or `{"tasks": "one per line"}` | Re-embeds and returns the saved list. |

## Notes

- Models load lazily inside the service thread, so the Flask process starts
  fast — but you'll see `loading_models` in the UI for ~10–20 seconds on first
  launch.
- The Resemblyzer embedding is held only in memory. Restart the server and
  you'll need to re-enroll. (Persisting to `data/enrolled.npy` is a 5-line
  change if you want it.)
- The VAD state machine uses hysteresis (`VAD_ON > VAD_OFF`) to avoid
  flickering on borderline frames.
