"""VoiceService — mic loop in a background thread, holds latest state.

Mirrors GazeService: detector.py is pure, this file does I/O + state. Other
modules call snapshot(); HTTP layer exposes it as /api/voice/state.

State machine (per VAD frame, ~32 ms):
    IDLE      → speech_prob ≥ VAD_ON      ⇒ open utterance, accumulate
    SPEAKING  → speech_prob <  VAD_OFF    ⇒ start tail-silence timer
              tail_silence ≥ TAIL_SILENCE ⇒ close utterance, process
              duration   ≥ MAX_UTTERANCE  ⇒ force-close (safety bound)

On close, only utterances ≥ MIN_UTTERANCE_MS get speaker-verified and
transcribed; below that we emit a TOO_SHORT verdict and move on.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from collections import deque
from typing import Optional

import numpy as np

from . import detector

SR = detector.SAMPLE_RATE
FRAME = detector.VAD_FRAME_SAMPLES

# Tunables — all env-overridable
MIN_UTTERANCE_MS    = int(os.environ.get("VOICE_MIN_UTTERANCE_MS", "5000"))
HISTORY_LIMIT       = int(os.environ.get("VOICE_HISTORY_LIMIT", "25"))
TAIL_SILENCE_MS     = int(os.environ.get("VOICE_TAIL_SILENCE_MS", "700"))
MAX_UTTERANCE_MS    = int(os.environ.get("VOICE_MAX_UTTERANCE_MS", "20000"))
VAD_ON              = float(os.environ.get("VOICE_VAD_ON",  "0.55"))
VAD_OFF             = float(os.environ.get("VOICE_VAD_OFF", "0.35"))
SPEAKER_THRESHOLD   = float(os.environ.get("VOICE_SPEAKER_THRESHOLD", "0.75"))
RELEVANCE_THRESHOLD = float(os.environ.get("VOICE_RELEVANCE_THRESHOLD", "0.40"))
WHISPER_MODEL       = os.environ.get("VOICE_WHISPER_MODEL", "base.en")
ENROLL_SECONDS      = int(os.environ.get("VOICE_ENROLL_SECONDS", "6"))


class VoiceService:
    def __init__(self, mic_index: Optional[int] = None):
        self.mic_index = mic_index
        self._thread: Optional[threading.Thread] = None
        self._worker: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        # Decouples mic capture from heavy processing (speaker/ASR/embed).
        # Mic thread enqueues utterances; worker thread drains the queue.
        self._work_q: "queue.Queue" = queue.Queue()
        self._history: deque = deque(maxlen=HISTORY_LIMIT)

        # Lazy heavy models — created inside the service thread
        self._vad = None
        self._spk = None
        self._asr = None
        self._txt = None

        # User-supplied state
        self._enrolled_emb: Optional[np.ndarray] = None
        self._task_texts: list[str] = []
        self._task_embs: np.ndarray = np.zeros((0, detector.TEXT_EMBED_DIM), dtype=np.float32)
        self._context_dirty = False

        # Enrollment flow (set by HTTP thread, consumed by service thread)
        self._enroll_request = False
        self._enroll_active = False
        self._enroll_progress = 0.0

        # Snapshot exposed to HTTP
        self._state: dict = {
            "speaking": False,
            "current_utterance_ms": 0,
            "reason": "not_started",
            "last": None,
            "queue_depth": 0,
            "worker_busy": False,
        }

    # ---------- public API (called from HTTP thread) ----------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(
            target=self._worker_loop, daemon=True, name="VoiceWorker"
        )
        self._worker.start()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="VoiceService"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._work_q.put(None)  # unblock worker
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        if self._worker:
            self._worker.join(timeout=5)
            self._worker = None

    def snapshot(self) -> dict:
        with self._lock:
            s = dict(self._state)
            s["enrolled"] = self._enrolled_emb is not None
            s["context_count"] = len(self._task_texts)
            s["enroll_active"] = self._enroll_active
            s["enroll_progress"] = round(self._enroll_progress, 2)
            s["history"] = list(self._history)[::-1]  # newest first
            s["thresholds"] = {
                "min_utterance_ms": MIN_UTTERANCE_MS,
                "tail_silence_ms": TAIL_SILENCE_MS,
                "speaker": SPEAKER_THRESHOLD,
                "relevance": RELEVANCE_THRESHOLD,
            }
            return s

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()
            self._state["last"] = None

    def request_enroll(self) -> None:
        self._enroll_request = True

    def reset_enroll(self) -> None:
        with self._lock:
            self._enrolled_emb = None

    def set_context(self, tasks: list[str]) -> None:
        cleaned = [t.strip() for t in tasks if t and t.strip()]
        with self._lock:
            self._task_texts = cleaned
            self._context_dirty = True

    def get_context(self) -> list[str]:
        with self._lock:
            return list(self._task_texts)

    # ---------- main loop ----------

    def _loop(self) -> None:
        try:
            import sounddevice as sd
        except Exception as e:
            self._set_reason(f"sounddevice_unavailable:{e}")
            return

        self._set_reason("loading_models")
        try:
            self._vad = detector.make_vad()
            self._spk = detector.make_speaker_encoder()
            self._asr = detector.make_asr(WHISPER_MODEL)
            self._txt = detector.make_text_encoder()
        except Exception as e:
            self._set_reason(f"model_load_failed:{e}")
            return

        # Embed any context that was set before models finished loading
        self._refresh_context_if_dirty()
        self._set_reason("idle")

        try:
            stream = sd.InputStream(
                samplerate=SR, channels=1, dtype="float32",
                blocksize=FRAME, device=self.mic_index,
            )
            stream.start()
        except Exception as e:
            self._set_reason(f"mic_open_failed:{e}")
            return

        in_speech = False
        utt_buf: list[np.ndarray] = []
        utt_start_ms = 0.0
        silence_run_ms = 0.0
        ms_per_frame = 1000.0 * FRAME / SR

        try:
            while not self._stop.is_set():
                # Handle context updates + enrollment requests between frames
                self._refresh_context_if_dirty()
                if self._enroll_request:
                    self._enroll_request = False
                    self._do_enrollment(stream)
                    continue

                data, _ = stream.read(FRAME)
                frame = data[:, 0].copy()
                prob = detector.vad_prob(self._vad, frame)
                now_ms = time.time() * 1000.0

                if not in_speech:
                    if prob >= VAD_ON:
                        in_speech = True
                        utt_buf = [frame]
                        utt_start_ms = now_ms
                        silence_run_ms = 0.0
                        with self._lock:
                            self._state["speaking"] = True
                            self._state["current_utterance_ms"] = 0
                            self._state["reason"] = "speaking"
                else:
                    utt_buf.append(frame)
                    cur_ms = int(now_ms - utt_start_ms)
                    silence_run_ms = silence_run_ms + ms_per_frame if prob < VAD_OFF else 0.0
                    with self._lock:
                        self._state["current_utterance_ms"] = cur_ms

                    if silence_run_ms >= TAIL_SILENCE_MS or cur_ms >= MAX_UTTERANCE_MS:
                        in_speech = False
                        utt = np.concatenate(utt_buf)
                        # Hand off to worker — mic loop keeps capturing immediately.
                        self._work_q.put((utt, cur_ms))
                        with self._lock:
                            self._state["speaking"] = False
                            self._state["current_utterance_ms"] = 0
                            self._state["queue_depth"] = self._work_q.qsize()
                            self._state["reason"] = "idle"
                        utt_buf = []
                        silence_run_ms = 0.0
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    # ---------- worker thread (heavy processing) ----------

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._work_q.get(timeout=0.25)
            except queue.Empty:
                continue
            if item is None:
                break
            samples, duration_ms = item
            with self._lock:
                self._state["worker_busy"] = True
                self._state["queue_depth"] = self._work_q.qsize()
            try:
                self._process_utterance(samples, duration_ms)
            except Exception as e:
                with self._lock:
                    self._history.append({
                        "duration_ms": duration_ms,
                        "is_user": False,
                        "speaker_score": None,
                        "transcript": f"[worker_error:{e}]",
                        "relevance": 0.0,
                        "verdict": "ERROR",
                        "at": time.time(),
                    })
            finally:
                with self._lock:
                    self._state["worker_busy"] = False
                    self._state["queue_depth"] = self._work_q.qsize()

    # ---------- helpers (service thread only) ----------

    def _set_reason(self, reason: str) -> None:
        with self._lock:
            self._state["reason"] = reason

    def _refresh_context_if_dirty(self) -> None:
        with self._lock:
            if not self._context_dirty or self._txt is None:
                return
            texts = list(self._task_texts)
            self._context_dirty = False
        embs = (
            detector.text_embeddings(self._txt, texts)
            if texts else np.zeros((0, detector.TEXT_EMBED_DIM), dtype=np.float32)
        )
        with self._lock:
            self._task_embs = embs

    def _do_enrollment(self, stream) -> None:
        total = SR * ENROLL_SECONDS
        captured = 0
        chunks: list[np.ndarray] = []
        self._enroll_active = True
        self._enroll_progress = 0.0
        with self._lock:
            self._state["reason"] = "enrolling"
        try:
            while captured < total and not self._stop.is_set():
                data, _ = stream.read(FRAME)
                chunks.append(data[:, 0].copy())
                captured += FRAME
                self._enroll_progress = captured / total
            wav = np.concatenate(chunks).astype(np.float32)
            try:
                emb = detector.speaker_embedding(self._spk, wav)
                with self._lock:
                    self._enrolled_emb = emb
                    self._state["reason"] = "idle"
            except Exception as e:
                self._set_reason(f"enroll_failed:{e}")
        finally:
            self._enroll_active = False
            self._enroll_progress = 0.0

    def _process_utterance(self, samples: np.ndarray, duration_ms: int) -> None:
        from concurrent.futures import ThreadPoolExecutor

        is_user = True
        speaker_score: Optional[float] = None
        transcript = ""
        rel = 0.0

        long_enough = duration_ms >= MIN_UTTERANCE_MS

        # Speaker verification + ASR are independent — run in parallel so total
        # wall time is max(spk, asr) instead of spk + asr.
        if long_enough:
            with ThreadPoolExecutor(max_workers=2) as pool:
                spk_fut = (
                    pool.submit(detector.speaker_embedding, self._spk, samples)
                    if self._enrolled_emb is not None else None
                )
                asr_fut = pool.submit(detector.transcribe, self._asr, samples)

                if spk_fut is not None:
                    try:
                        emb = spk_fut.result()
                        speaker_score = detector.cosine(emb, self._enrolled_emb)
                        is_user = speaker_score >= SPEAKER_THRESHOLD
                    except Exception:
                        pass

                try:
                    transcript = asr_fut.result()
                except Exception as e:
                    transcript = f"[asr_error:{e}]"

            if is_user and transcript:
                with self._lock:
                    task_embs = self._task_embs if self._task_embs.shape[0] > 0 else None
                if task_embs is not None:
                    utt_emb = detector.text_embeddings(self._txt, [transcript])[0]
                    rel = detector.relevance(utt_emb, task_embs)

        with self._lock:
            has_context = self._task_embs.shape[0] > 0

        verdict = detector.classify(
            duration_ms=duration_ms,
            is_user=is_user,
            relevance_score=rel,
            has_context=has_context,
            min_utterance_ms=MIN_UTTERANCE_MS,
            relevance_threshold=RELEVANCE_THRESHOLD,
        )

        last = {
            "duration_ms": duration_ms,
            "is_user": is_user,
            "speaker_score": round(speaker_score, 3) if speaker_score is not None else None,
            "transcript": transcript,
            "relevance": round(rel, 3),
            "verdict": verdict,
            "at": time.time(),
        }
        with self._lock:
            self._state["last"] = last
            self._history.append(last)
