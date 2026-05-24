"""Engine entry point — module-level singleton for the voice service."""

import os
from typing import Optional

from backend.distraction.voice.service import VoiceService

_voice: Optional[VoiceService] = None


def get_voice_service() -> VoiceService:
    global _voice
    if _voice is None:
        idx = os.environ.get("MIC_INDEX")
        _voice = VoiceService(mic_index=int(idx) if idx else None)
    return _voice
