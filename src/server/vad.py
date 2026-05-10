"""
Voice Activity Detection module with segment tracking.

Enhanced from upstream to support:
- Segment detection with speech_start / speech_end timestamps
- process_chunk() returning VADEvent enum (SILENCE, SPEECH_START, SPEECH_CONTINUE, SPEECH_END)
- Configurable thresholds for speech detection and silence durations
- Silero VAD -> webrtcvad -> energy-based fallback chain
"""

import enum
import time
import struct
from typing import Optional

import numpy as np
from loguru import logger


class VADEvent(enum.Enum):
    """Result of processing an audio chunk through VAD."""
    SILENCE = "silence"
    SPEECH_START = "speech_start"
    SPEECH_CONTINUE = "speech_continue"
    SPEECH_END = "speech_end"


class VoiceActivityDetector:
    """
    Voice Activity Detection with segment tracking.

    Maintains internal state to detect when speech starts and ends,
    enabling upstream code to accumulate speech segments and flush
    them at appropriate boundaries.
    """

    def __init__(
        self,
        speech_threshold: float = 0.5,
        silence_duration_ms_flush: int = 300,
        silence_duration_ms_turn_end: int = 1500,
        sample_rate: int = 16000,
    ):
        self.speech_threshold = speech_threshold
        self.silence_duration_ms_flush = silence_duration_ms_flush
        self.silence_duration_ms_turn_end = silence_duration_ms_turn_end
        self.sample_rate = sample_rate

        # Internal state for segment tracking
        self._in_speech = False
        self._speech_start_time: Optional[float] = None
        self._last_speech_time: Optional[float] = None
        self._silence_start_time: Optional[float] = None

        # VAD backend
        self._backend = "none"
        self._silero_model = None
        self._webrtc_vad = None

        self._load_model()

    def _load_model(self):
        """Load VAD model with fallback chain: Silero -> webrtcvad -> energy."""
        # Try Silero VAD first
        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
            )
            self._silero_model = model
            self._backend = "silero"
            logger.info("Silero VAD loaded")
            return
        except Exception as e:
            logger.warning(f"Silero VAD not available: {e}")

        # Try webrtcvad
        try:
            import webrtcvad
            self._webrtc_vad = webrtcvad.Vad()
            self._webrtc_vad.set_mode(2)  # 0=least aggressive, 3=most aggressive
            self._backend = "webrtcvad"
            logger.info("webrtcvad loaded")
            return
        except Exception as e:
            logger.warning(f"webrtcvad not available: {e}")

        # Fall back to energy-based detection
        self._backend = "energy"
        logger.warning("Using energy-based VAD fallback (no ML model)")

    def _get_speech_probability(self, audio: np.ndarray) -> float:
        """
        Get the probability that the audio chunk contains speech.

        Returns a float between 0.0 and 1.0.
        """
        if self._backend == "silero":
            return self._silero_probability(audio)
        elif self._backend == "webrtcvad":
            return self._webrtcvad_probability(audio)
        else:
            return self._energy_probability(audio)

    def _silero_probability(self, audio: np.ndarray) -> float:
        """Get speech probability from Silero VAD."""
        try:
            import torch
            audio_tensor = torch.from_numpy(audio).float()
            if audio_tensor.dim() > 1:
                audio_tensor = audio_tensor.squeeze()
            speech_prob = self._silero_model(audio_tensor, self.sample_rate).item()
            return speech_prob
        except Exception as e:
            logger.error(f"Silero VAD error: {e}")
            return 0.0

    def _webrtcvad_probability(self, audio: np.ndarray) -> float:
        """Get speech probability from webrtcvad (binary: 0.0 or 1.0)."""
        try:
            # webrtcvad requires 16-bit PCM at 8000, 16000, 32000, or 48000 Hz
            # and frames of 10, 20, or 30 ms
            audio_int16 = (audio * 32767).astype(np.int16)
            raw_bytes = audio_int16.tobytes()

            # webrtcvad needs exact frame sizes: 10ms, 20ms, or 30ms
            frame_duration_ms = 30
            frame_size = int(self.sample_rate * frame_duration_ms / 1000)
            frame_bytes = frame_size * 2  # 2 bytes per int16 sample

            if len(raw_bytes) < frame_bytes:
                # Pad with silence if chunk is too small
                raw_bytes = raw_bytes + b"\x00" * (frame_bytes - len(raw_bytes))

            # Process multiple frames and average
            speech_frames = 0
            total_frames = 0
            offset = 0
            while offset + frame_bytes <= len(raw_bytes):
                frame = raw_bytes[offset : offset + frame_bytes]
                is_speech = self._webrtc_vad.is_speech(frame, self.sample_rate)
                if is_speech:
                    speech_frames += 1
                total_frames += 1
                offset += frame_bytes

            if total_frames == 0:
                return 0.0
            return speech_frames / total_frames
        except Exception as e:
            logger.error(f"webrtcvad error: {e}")
            return 0.0

    def _energy_probability(self, audio: np.ndarray) -> float:
        """
        Energy-based speech detection fallback.

        Computes RMS energy and maps it to a 0..1 probability using a
        simple sigmoid-like curve calibrated so that typical speech
        (~0.02-0.1 RMS on float32 [-1,1] audio) maps above 0.5 and
        background noise (~0.001-0.005) maps well below.
        """
        if len(audio) == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        # Tuned so rms ~0.01 -> ~0.5 probability
        # Using a logistic function: 1 / (1 + exp(-k*(rms - midpoint)))
        midpoint = 0.01
        steepness = 400.0
        exponent = -steepness * (rms - midpoint)
        # Clamp to avoid overflow
        exponent = max(min(exponent, 500.0), -500.0)
        import math
        probability = 1.0 / (1.0 + math.exp(exponent))
        return probability

    def is_speech(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """
        Simple binary check: does this audio contain speech?

        Preserved for backward compatibility with upstream callers.
        """
        old_sr = self.sample_rate
        self.sample_rate = sample_rate
        prob = self._get_speech_probability(audio)
        self.sample_rate = old_sr
        return prob > self.speech_threshold

    def process_chunk(self, audio: np.ndarray) -> VADEvent:
        """
        Process an audio chunk and return the current VAD state transition.

        Maintains internal state to track ongoing speech segments. Call this
        sequentially with consecutive audio chunks to get segment boundaries.

        Returns:
            VADEvent.SILENCE        - no speech detected, not in a speech segment
            VADEvent.SPEECH_START   - speech just started (first speech frame after silence)
            VADEvent.SPEECH_CONTINUE - speech is ongoing
            VADEvent.SPEECH_END     - silence exceeded the flush threshold after speech
        """
        now = time.monotonic()
        prob = self._get_speech_probability(audio)
        is_speech_now = prob > self.speech_threshold

        if is_speech_now:
            self._last_speech_time = now
            self._silence_start_time = None

            if not self._in_speech:
                # Transition: silence -> speech
                self._in_speech = True
                self._speech_start_time = now
                return VADEvent.SPEECH_START
            else:
                return VADEvent.SPEECH_CONTINUE
        else:
            # Current chunk is silence
            if self._in_speech:
                # We were in speech; track how long silence has lasted
                if self._silence_start_time is None:
                    self._silence_start_time = now

                silence_elapsed_ms = (now - self._silence_start_time) * 1000.0

                if silence_elapsed_ms >= self.silence_duration_ms_flush:
                    # Enough silence to end the speech segment
                    self._in_speech = False
                    speech_start = self._speech_start_time
                    self._speech_start_time = None
                    self._silence_start_time = None
                    return VADEvent.SPEECH_END
                else:
                    # Brief silence within speech -- treat as continuation
                    return VADEvent.SPEECH_CONTINUE
            else:
                return VADEvent.SILENCE

    def is_turn_end(self) -> bool:
        """
        Check whether silence has lasted long enough to indicate the speaker's
        turn is complete (i.e. they are done talking, not just pausing).

        Call this after process_chunk returns SPEECH_END or SILENCE to decide
        whether to finalize the full utterance for the AI backend.
        """
        if self._in_speech:
            return False
        if self._last_speech_time is None:
            return False
        elapsed_ms = (time.monotonic() - self._last_speech_time) * 1000.0
        return elapsed_ms >= self.silence_duration_ms_turn_end

    def reset(self):
        """Reset all internal state (e.g. when starting a new session)."""
        self._in_speech = False
        self._speech_start_time = None
        self._last_speech_time = None
        self._silence_start_time = None
        # Reset Silero hidden state if applicable
        if self._backend == "silero" and self._silero_model is not None:
            try:
                self._silero_model.reset_states()
            except Exception:
                pass

    @property
    def backend(self) -> str:
        """Return which VAD backend is active."""
        return self._backend

    @property
    def in_speech(self) -> bool:
        """Whether we are currently inside a speech segment."""
        return self._in_speech

    @property
    def speech_start_time(self) -> Optional[float]:
        """Monotonic timestamp when the current speech segment started, or None."""
        return self._speech_start_time

    @property
    def last_speech_time(self) -> Optional[float]:
        """Monotonic timestamp of the most recent speech frame, or None."""
        return self._last_speech_time
