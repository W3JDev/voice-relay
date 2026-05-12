"""
Text-to-Speech module using Kokoro (via Speaches/VSaaS) or Edge TTS fallback.

Phase 1 Voice Relay: Simplified TTS stack.
  Primary:  Kokoro via Speaches/VSaaS (OpenAI-compatible /v1/audio/speech)
  Fallback: Edge TTS (free Microsoft TTS, MP3 -> PCM conversion)
  Fallback: Mock (silence)
"""

import asyncio
import io
import os
import struct
from typing import Optional, AsyncGenerator

from loguru import logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 24000          # 24 kHz
SAMPLE_WIDTH = 2             # 16-bit (2 bytes per sample)
SILENCE_DURATION_S = 0.5     # Mock silence length


class RelayTTS:
    """Text-to-Speech for the Phase 1 Voice Relay.

    Backends (tried in order):
        1. Kokoro via Speaches / VSaaS gateway  (env: VSAAS_URL, VSAAS_API_KEY)
        2. Edge TTS  (pip install edge-tts; no credentials needed)
        3. Mock  (returns silence)
    """

    def __init__(
        self,
        voice: Optional[str] = None,
        edge_voice: Optional[str] = None,
    ):
        """
        Args:
            voice:      Kokoro voice name (default ``af_heart``).
            edge_voice: Edge TTS voice name (default ``en-US-AriaNeural``).
        """
        self._voice = voice or os.environ.get("KOKORO_VOICE", "af_heart")
        self._edge_voice = edge_voice or os.environ.get("EDGE_TTS_VOICE", "en-US-AriaNeural")
        self._kokoro_model = os.environ.get(
            "KOKORO_MODEL", "speaches-ai/Kokoro-82M-v1.0-ONNX"
        )

        self._backend: str = "mock"
        self._vsaas_client = None  # openai.OpenAI (sync, used in threads)

        self._init_backend()

    # ------------------------------------------------------------------
    # Backend initialisation
    # ------------------------------------------------------------------
    def _init_backend(self) -> None:
        """Probe available backends and lock in the first that works."""

        # --- 1. Kokoro via VSaaS / Speaches ---
        vsaas_url = os.environ.get("VSAAS_URL")
        vsaas_api_key = os.environ.get("VSAAS_API_KEY", "not-needed")

        if vsaas_url:
            try:
                from openai import OpenAI  # noqa: F811

                self._vsaas_client = OpenAI(
                    base_url=f"{vsaas_url.rstrip('/')}/v1",
                    api_key=vsaas_api_key,
                )
                self._backend = "kokoro"
                logger.info(
                    "TTS backend: Kokoro via VSaaS  "
                    f"(url={vsaas_url}, voice={self._voice})"
                )
                return
            except ImportError:
                logger.warning(
                    "openai package not installed -- cannot use Kokoro/VSaaS backend"
                )
            except Exception as exc:
                logger.warning(f"Kokoro/VSaaS init failed: {exc}")

        # --- 2. Edge TTS ---
        try:
            import edge_tts as _  # noqa: F401

            self._backend = "edge_tts"
            logger.info(
                f"TTS backend: Edge TTS  (voice={self._edge_voice})"
            )
            return
        except ImportError:
            logger.warning("edge-tts package not installed")

        # --- 3. Mock ---
        self._backend = "mock"
        logger.warning("TTS backend: Mock (silence) -- no TTS engine available")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def backend(self) -> str:
        """Return the name of the active backend."""
        return self._backend

    async def synthesize_stream(
        self, text: str, voice: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        """Stream synthesised audio as raw PCM chunks.

        Args:
            text:  The text to synthesise.
            voice: Optional Kokoro voice ID override (e.g. ``"am_adam"``).
                   If ``None``, uses the instance default (``self._voice``).

        Yields:
            ``bytes`` -- raw PCM audio (24 kHz, 16-bit signed integer, mono).
        """
        if self._backend == "kokoro":
            async for chunk in self._stream_kokoro(text, voice=voice):
                yield chunk

        elif self._backend == "edge_tts":
            async for chunk in self._stream_edge_tts(text):
                yield chunk

        else:
            # Mock -- yield a short block of silence
            yield self._generate_silence()

    # ------------------------------------------------------------------
    # Kokoro / VSaaS
    # ------------------------------------------------------------------
    async def _stream_kokoro(
        self, text: str, voice: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        """Request PCM audio from the VSaaS gateway and yield chunks."""
        loop = asyncio.get_running_loop()
        effective_voice = voice or self._voice

        try:
            # The OpenAI SDK's audio.speech.create returns an HttpxBinaryResponseContent.
            # We run the synchronous call in a thread so we don't block the event loop.
            response = await loop.run_in_executor(
                None,
                self._kokoro_request,
                text,
                effective_voice,
            )

            # Yield larger chunks to reduce WebSocket overhead and give the
            # client ring-buffer more contiguous audio per write.
            # 48000 bytes = ~1 second at 24 kHz 16-bit mono.
            for chunk in response.iter_bytes(chunk_size=48000):
                if chunk:
                    yield chunk

        except Exception as exc:
            logger.error(f"Kokoro TTS error: {exc}")
            yield self._generate_silence()

    def _kokoro_request(self, text: str, voice: Optional[str] = None):
        """Blocking call -- meant to be run via ``run_in_executor``."""
        effective_voice = voice or self._voice
        response = self._vsaas_client.audio.speech.create(
            model=self._kokoro_model,
            voice=effective_voice,
            input=text,
            response_format="pcm",
            speed=0.95,  # Slightly slower for a calm, broadcast-quality pace
            extra_body={"sample_rate": SAMPLE_RATE},
        )
        return response

    # ------------------------------------------------------------------
    # Edge TTS
    # ------------------------------------------------------------------
    async def _stream_edge_tts(self, text: str) -> AsyncGenerator[bytes, None]:
        """Use Edge TTS to produce MP3, decode to PCM, and yield chunks."""
        import edge_tts

        try:
            communicate = edge_tts.Communicate(text, self._edge_voice)

            # Collect MP3 bytes -- edge-tts streams small MP3 frames.
            mp3_buf = io.BytesIO()
            async for msg in communicate.stream():
                if msg["type"] == "audio":
                    mp3_buf.write(msg["data"])

            mp3_buf.seek(0)

            # Decode MP3 -> raw PCM 24 kHz 16-bit mono
            pcm_data = await asyncio.get_running_loop().run_in_executor(
                None, self._mp3_to_pcm, mp3_buf.read()
            )

            # Yield in reasonable-sized chunks (~100 ms)
            chunk_size = SAMPLE_RATE * SAMPLE_WIDTH // 10  # 4800 bytes = 100 ms
            offset = 0
            while offset < len(pcm_data):
                yield pcm_data[offset : offset + chunk_size]
                offset += chunk_size

        except Exception as exc:
            logger.error(f"Edge TTS error: {exc}")
            yield self._generate_silence()

    @staticmethod
    def _mp3_to_pcm(mp3_bytes: bytes) -> bytes:
        """Decode MP3 bytes to raw PCM (24 kHz, 16-bit signed, mono).

        Uses pydub if available, otherwise falls back to a minimal
        subprocess call to ffmpeg.
        """
        try:
            from pydub import AudioSegment

            seg = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
            seg = seg.set_frame_rate(SAMPLE_RATE).set_sample_width(SAMPLE_WIDTH).set_channels(1)
            return seg.raw_data
        except ImportError:
            pass

        # Fallback: ffmpeg via subprocess
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
            tmp.write(mp3_bytes)
            tmp.flush()
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", tmp.name,
                    "-f", "s16le",
                    "-ar", str(SAMPLE_RATE),
                    "-ac", "1",
                    "-",
                ],
                capture_output=True,
            )
            if result.returncode != 0:
                logger.error(f"ffmpeg decode failed: {result.stderr.decode(errors='replace')}")
                return b""
            return result.stdout

    # ------------------------------------------------------------------
    # Mock
    # ------------------------------------------------------------------
    @staticmethod
    def _generate_silence() -> bytes:
        """Return a short block of silence as raw PCM bytes."""
        num_samples = int(SAMPLE_RATE * SILENCE_DURATION_S)
        return struct.pack(f"<{num_samples}h", *([0] * num_samples))
