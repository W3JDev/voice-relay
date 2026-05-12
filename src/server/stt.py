"""
Speech-to-Text module with VSaaS/Speaches API support and local Whisper fallback.

Enhanced from upstream to support:
- transcribe_via_api(): call VSaaS/Speaches /v1/audio/transcriptions via OpenAI SDK
- transcribe_segment(): efficient handling of small audio chunks
- Cascading fallback: API -> local faster-whisper -> local openai-whisper -> empty string
"""

import asyncio
import io
import tempfile
import wave
from typing import Optional

import numpy as np
from loguru import logger


class WhisperSTT:
    """Speech-to-Text with API-first and local Whisper fallback."""

    def __init__(
        self,
        model_name: str = "base",
        device: str = "auto",
        language: str = "en",
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.language = language
        self.api_url = api_url
        self.api_key = api_key
        self.model = None
        self._backend = "mock"
        self._api_client = None

        self._init_api_client()
        self._load_model()

    def _init_api_client(self):
        """Initialize the OpenAI-compatible API client for VSaaS/Speaches."""
        if not self.api_url or not self.api_key:
            logger.info("No VSaaS API URL/key configured; API transcription disabled")
            return

        try:
            from openai import OpenAI
            self._api_client = OpenAI(
                base_url=f"{self.api_url.rstrip('/')}/v1",
                api_key=self.api_key,
            )
            logger.info(f"VSaaS/Speaches API client initialized: {self.api_url}")
        except Exception as e:
            logger.warning(f"Failed to initialize API client: {e}")
            self._api_client = None

    def _load_model(self):
        """Load local Whisper model as fallback."""
        # Try faster-whisper first
        try:
            from faster_whisper import WhisperModel

            if self.device == "auto":
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                    compute_type = "float16"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.device = "cpu"
                    compute_type = "int8"
                else:
                    self.device = "cpu"
                    compute_type = "int8"
            elif self.device == "cuda":
                compute_type = "float16"
            else:
                compute_type = "int8"

            logger.info(f"Loading faster-whisper {self.model_name} on {self.device}")
            self.model = WhisperModel(
                self.model_name,
                device=self.device if self.device != "mps" else "cpu",
                compute_type=compute_type,
            )
            self._backend = "faster-whisper"
            logger.info("faster-whisper loaded")
            return
        except ImportError:
            logger.warning("faster-whisper not available")
        except Exception as e:
            logger.warning(f"faster-whisper failed: {e}")

        # Try openai-whisper
        try:
            import whisper

            if self.device == "auto":
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(f"Loading openai-whisper {self.model_name}")
            self.model = whisper.load_model(self.model_name, device=self.device)
            self._backend = "openai-whisper"
            logger.info("openai-whisper loaded")
            return
        except ImportError:
            logger.warning("openai-whisper not available")
        except Exception as e:
            logger.warning(f"openai-whisper failed: {e}")

        # Mock mode for testing
        if self._api_client:
            logger.info("No local STT backend; will use API only")
            self._backend = "api-only"
        else:
            logger.warning("No STT backend available - using mock mode")
            self._backend = "mock"

    @staticmethod
    def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
        """Convert a float32 numpy audio array to WAV bytes in memory."""
        audio_int16 = (audio * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        buf.seek(0)
        return buf.read()

    def transcribe_via_api(
        self,
        audio: np.ndarray,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        sample_rate: int = 16000,
    ) -> str:
        """
        Transcribe audio via the VSaaS/Speaches /v1/audio/transcriptions endpoint.

        Uses the OpenAI SDK pointed at the Speaches-compatible server.

        Args:
            audio: float32 numpy array of audio samples
            api_url: Override API URL (uses instance default if None)
            api_key: Override API key (uses instance default if None)
            sample_rate: Sample rate of the audio (default 16000)

        Returns:
            Transcribed text, or empty string on failure.
        """
        effective_url = api_url or self.api_url
        effective_key = api_key or self.api_key

        if not effective_url or not effective_key:
            logger.debug("API transcription skipped: no URL/key")
            return ""

        # Build a client (reuse instance client if args match defaults)
        if api_url or api_key:
            from openai import OpenAI
            client = OpenAI(
                base_url=f"{effective_url.rstrip('/')}/v1",
                api_key=effective_key,
            )
        elif self._api_client is not None:
            client = self._api_client
        else:
            return ""

        wav_bytes = self._audio_to_wav_bytes(audio, sample_rate)

        try:
            audio_file = io.BytesIO(wav_bytes)
            audio_file.name = "audio.wav"
            transcript = client.audio.transcriptions.create(
                model="Systran/faster-whisper-base",
                file=audio_file,
            )
            text = transcript.text.strip() if transcript.text else ""
            logger.debug(f"API transcription ({len(audio)} samples): {text[:80]}")
            return text
        except Exception as e:
            logger.warning(f"API transcription failed: {e}")
            return ""

    def _transcribe_local(self, audio: np.ndarray) -> str:
        """Transcribe using the local Whisper model."""
        if self._backend == "faster-whisper":
            segments, _info = self.model.transcribe(
                audio,
                language=self.language,
                beam_size=5,
                vad_filter=True,
            )
            return " ".join(segment.text for segment in segments).strip()

        elif self._backend == "openai-whisper":
            result = self.model.transcribe(audio, language=self.language)
            return result["text"].strip()

        else:
            return ""

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        """
        Synchronous transcription with cascading fallback.

        Order: API -> local Whisper -> empty string.
        """
        # Try API first
        if self._api_client is not None:
            text = self.transcribe_via_api(audio)
            if text:
                return text
            logger.debug("API returned empty; falling back to local")

        # Try local Whisper
        if self._backend in ("faster-whisper", "openai-whisper"):
            try:
                text = self._transcribe_local(audio)
                if text:
                    return text
            except Exception as e:
                logger.error(f"Local transcription error: {e}")

        # Mock fallback -- return empty so fake text doesn't reach the LLM
        if self._backend == "mock":
            logger.debug(f"Mock STT: received {len(audio)} samples (no real backend)")
            return ""

        return ""

    async def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe audio to text (async wrapper).

        Tries API first (if configured), then local Whisper, then returns
        empty string.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, audio)

    async def transcribe_segment(self, audio_segment: np.ndarray) -> str:
        """
        Transcribe a small audio segment efficiently.

        Optimized for short speech segments (typically 0.3-5 seconds) produced
        by VAD-based segmentation. Skips segments that are too short to contain
        meaningful speech.

        Args:
            audio_segment: float32 numpy array of a single speech segment

        Returns:
            Transcribed text or empty string.
        """
        # Skip extremely short segments (< 200ms at 16kHz = 3200 samples)
        min_samples = 3200
        if len(audio_segment) < min_samples:
            logger.debug(
                f"Segment too short ({len(audio_segment)} samples < {min_samples}); skipping"
            )
            return ""

        # Skip near-silent segments
        rms = float(np.sqrt(np.mean(audio_segment.astype(np.float64) ** 2)))
        if rms < 0.005:
            logger.debug(f"Segment too quiet (rms={rms:.4f}); skipping")
            return ""

        # For short segments, prefer the API (lower latency than loading local model)
        if self._api_client is not None:
            text = await asyncio.get_event_loop().run_in_executor(
                None, self.transcribe_via_api, audio_segment
            )
            if text:
                return text

        # Fall through to full transcription pipeline
        return await self.transcribe(audio_segment)

    @property
    def backend(self) -> str:
        """Return which STT backend is active."""
        return self._backend

    @property
    def has_api(self) -> bool:
        """Whether an API client is configured and available."""
        return self._api_client is not None
