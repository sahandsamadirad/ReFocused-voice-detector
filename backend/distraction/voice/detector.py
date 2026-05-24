"""Pure voice analysis — frames in, verdicts out. No mic, no threads.

Each helper is independent and stateless (or holds only model weights), so
service.py can compose them. Heavy imports happen inside factory functions
so module import stays cheap.

Pipeline:
    raw mic frame (float32, 16 kHz)
        → vad_prob()                  speech probability
    on utterance close:
        → speaker_embedding() + cosine()      is_user?
        → transcribe()                        text
        → text_embeddings() + relevance()     on-task score
        → classify()                          ON_TASK / OFF_TASK / ...
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

SAMPLE_RATE = 16_000
VAD_FRAME_SAMPLES = 512  # Silero @ 16 kHz expects 512-sample chunks (32 ms)
TEXT_EMBED_DIM = 384     # all-MiniLM-L6-v2


# ---------- VAD (Silero) ------------------------------------------------

def make_vad():
    import torch
    from silero_vad import load_silero_vad
    torch.set_num_threads(1)
    return load_silero_vad()


def vad_prob(model, frame: np.ndarray) -> float:
    """frame: float32 mono samples at 16 kHz, length == VAD_FRAME_SAMPLES."""
    import torch
    with torch.no_grad():
        t = torch.from_numpy(frame).float()
        return float(model(t, SAMPLE_RATE).item())


# ---------- Speaker verification (Resemblyzer) --------------------------

def make_speaker_encoder():
    from resemblyzer import VoiceEncoder
    return VoiceEncoder(verbose=False)


def speaker_embedding(encoder, samples_float32: np.ndarray) -> np.ndarray:
    """samples_float32: mono 16 kHz, range [-1, 1]. Returns 256-D embedding."""
    from resemblyzer import preprocess_wav
    wav = preprocess_wav(samples_float32, source_sr=SAMPLE_RATE)
    return encoder.embed_utterance(wav)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)) + 1e-9
    nb = float(np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / (na * nb))


# ---------- ASR (faster-whisper) ----------------------------------------

def make_asr(model_name: str = "base.en"):
    from faster_whisper import WhisperModel
    return WhisperModel(model_name, device="cpu", compute_type="int8")


def transcribe(asr, samples_float32: np.ndarray) -> str:
    # VAD already gated us; tell whisper not to redo it.
    segments, _ = asr.transcribe(
        samples_float32,
        language="en",
        vad_filter=False,
        beam_size=1,
        condition_on_previous_text=False,
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


# ---------- Text embedding + semantic relevance -------------------------

def make_text_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def text_embeddings(encoder, texts: Iterable[str]) -> np.ndarray:
    return encoder.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def relevance(utt_emb: np.ndarray, task_embs: np.ndarray) -> float:
    """Max cosine over all task contexts. Inputs must already be L2-normalized."""
    if task_embs.size == 0:
        return 0.0
    return float(np.max(task_embs @ utt_emb))


# ---------- Verdict -----------------------------------------------------

def classify(
    duration_ms: int,
    is_user: bool,
    relevance_score: float,
    has_context: bool,
    min_utterance_ms: int,
    relevance_threshold: float,
) -> str:
    if duration_ms < min_utterance_ms:
        return "TOO_SHORT"
    if not is_user:
        return "NOT_USER"
    if not has_context:
        return "NO_CONTEXT"
    return "ON_TASK" if relevance_score >= relevance_threshold else "OFF_TASK"
