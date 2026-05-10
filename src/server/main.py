"""
Voice Relay Server -- Phase 1 Rewrite

Changes from upstream:
  - Per-session conversation history (no global singleton)
  - VAD-triggered streaming STT with partial/final transcripts
  - Barge-in support (interrupt TTS when user speaks)
  - Session lifecycle with reconnection and TTL-based cleanup
  - New environment variables (VSAAS_URL, VSAAS_API_KEY, VOICE_RELAY_SESSION_TTL)
  - GET /health endpoint
  - New message types: session_start, transcript, interrupted, reconnect, config
"""

import asyncio
import base64
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic_settings import BaseSettings

from .stt import WhisperSTT
from .tts import RelayTTS
from .backend import AIBackend
from .vad import VoiceActivityDetector
from .auth import token_manager, load_keys_from_env, APIKey
from .text_utils import clean_for_speech


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Server configuration with Phase 1 additions."""

    # Server
    host: str = "0.0.0.0"
    port: int = 8765

    # Auth
    require_auth: bool = False
    master_key: Optional[str] = None

    # STT
    stt_model: str = "base"
    stt_device: str = "auto"

    # TTS
    tts_model: str = "chatterbox"
    tts_voice: Optional[str] = None

    # AI Backend (kept for backward compatibility)
    backend_type: str = "openai"
    backend_url: str = "https://api.openai.com/v1"
    backend_model: str = "gpt-4o-mini"
    openai_api_key: Optional[str] = None

    # OpenClaw Gateway (kept for backward compatibility)
    openclaw_gateway_url: Optional[str] = None
    openclaw_gateway_token: Optional[str] = None

    # Audio
    sample_rate: int = 16000

    # --- Phase 1 additions ---

    # VSaaS gateway (external STT/TTS via Speaches)
    vsaas_url: Optional[str] = None
    vsaas_api_key: Optional[str] = None

    # Session TTL in seconds (default 1 hour)
    session_ttl: int = 3600

    class Config:
        env_prefix = "OPENCLAW_"
        env_file = ".env"
        # Allow VSAAS_* and VOICE_RELAY_* without OPENCLAW_ prefix as well.
        # We handle those manually in _load_phase1_env below.


def _load_phase1_env(settings: "Settings") -> "Settings":
    """
    Load Phase 1 environment variables that use non-OPENCLAW_ prefixes.

    Supports:
      VSAAS_URL, VSAAS_API_KEY, VOICE_RELAY_SESSION_TTL
    """
    if os.getenv("VSAAS_URL"):
        settings.vsaas_url = os.getenv("VSAAS_URL")
    if os.getenv("VSAAS_API_KEY"):
        settings.vsaas_api_key = os.getenv("VSAAS_API_KEY")
    ttl = os.getenv("VOICE_RELAY_SESSION_TTL")
    if ttl is not None:
        try:
            settings.session_ttl = int(ttl)
        except ValueError:
            logger.warning(
                f"Invalid VOICE_RELAY_SESSION_TTL value '{ttl}', using default"
            )
    return settings


settings = Settings()
settings = _load_phase1_env(settings)


# ---------------------------------------------------------------------------
# Per-session state
# ---------------------------------------------------------------------------

# VAD tuning constants
_SILENCE_FLUSH_MS = 300     # silence duration to flush segment to STT
_SILENCE_FINAL_MS = 1500    # silence duration to mark transcript as final
_SAMPLES_PER_MS_16K = 16    # 16 kHz -> 16 samples per ms


@dataclass
class SessionState:
    """Mutable state for a single voice session."""

    session_id: str

    # Audio accumulation
    audio_buffer: List[np.ndarray] = field(default_factory=list)

    # Conversation history (per-session, not global)
    conversation_history: List[Dict] = field(default_factory=list)

    # VAD state
    is_speech_active: bool = False
    silence_samples: int = 0
    speech_segment: List[np.ndarray] = field(default_factory=list)
    pending_transcript: str = ""

    # TTS / barge-in state
    is_responding: bool = False
    interrupted: bool = False
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)

    # Connection bookkeeping
    connected: bool = True
    websocket: Optional[WebSocket] = None
    last_activity: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)

    # Optional per-session backend overrides (via "config" message)
    backend_url_override: Optional[str] = None
    backend_model_override: Optional[str] = None

    def touch(self) -> None:
        """Update last-activity timestamp."""
        self.last_activity = time.time()

    def is_expired(self, ttl: int) -> bool:
        """Return True if the session has exceeded its TTL."""
        return (time.time() - self.last_activity) > ttl

    def reset_vad(self) -> None:
        """Reset VAD tracking between utterances."""
        self.is_speech_active = False
        self.silence_samples = 0
        self.speech_segment = []
        self.pending_transcript = ""


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(title="Voice Relay", version="1.0.0-phase1")

# Global instances (initialized on startup)
stt: Optional[WhisperSTT] = None
tts: Optional[RelayTTS] = None
default_backend: Optional[AIBackend] = None
vad: Optional[VoiceActivityDetector] = None

# Session registry
sessions: Dict[str, SessionState] = {}

# Uptime tracking
_server_start_time: float = 0.0

# Background cleanup task handle
_cleanup_task: Optional[asyncio.Task] = None


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    """Initialize models and background tasks on server start."""
    global stt, tts, default_backend, vad, _server_start_time, _cleanup_task

    _server_start_time = time.time()
    logger.info("Initializing Voice Relay server (Phase 1)...")

    # Load API keys
    load_keys_from_env()
    if settings.require_auth:
        logger.info("Authentication ENABLED")
    else:
        logger.warning("Authentication DISABLED (dev mode)")

    # Initialize STT
    logger.info(f"Loading STT model: {settings.stt_model}")
    stt = WhisperSTT(
        model_name=settings.stt_model,
        device=settings.stt_device,
    )

    # Initialize TTS
    logger.info(f"Loading TTS model: {settings.tts_model}")
    tts = RelayTTS(
        voice=settings.tts_voice,
    )

    # Initialize default AI backend
    default_backend = _create_backend()

    # Initialize VAD
    logger.info("Loading VAD model")
    vad = VoiceActivityDetector()

    # Start session cleanup loop
    _cleanup_task = asyncio.create_task(_session_cleanup_loop())

    logger.info("Voice Relay server ready")


@app.on_event("shutdown")
async def shutdown():
    """Cancel background tasks on shutdown."""
    if _cleanup_task and not _cleanup_task.done():
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            pass


def _create_backend(
    url: Optional[str] = None,
    model: Optional[str] = None,
) -> AIBackend:
    """
    Create an AIBackend instance.

    Uses OpenClaw gateway env vars if available, then falls back to
    the provided overrides (from a session config message) or the
    global settings.
    """
    gateway_url = settings.openclaw_gateway_url or os.getenv("OPENCLAW_GATEWAY_URL")
    gateway_token = settings.openclaw_gateway_token or os.getenv("OPENCLAW_GATEWAY_TOKEN")

    if gateway_url and gateway_token:
        logger.info(f"Connecting to OpenClaw gateway: {gateway_url}")
        return AIBackend(
            backend_type="openai",
            url=f"{gateway_url}/v1",
            model="openclaw:voice",
            api_key=gateway_token,
            system_prompt=(
                "This conversation is happening via real-time voice chat. "
                "Keep responses concise and conversational -- a few sentences "
                "at most unless the topic genuinely needs depth. "
                "No markdown, bullet points, code blocks, or special formatting."
            ),
        )

    return AIBackend(
        backend_type=settings.backend_type,
        url=url or settings.backend_url,
        model=model or settings.backend_model,
        api_key=settings.openai_api_key or os.getenv("OPENAI_API_KEY"),
    )


def _get_session_backend(session: SessionState) -> AIBackend:
    """
    Return a per-session backend if overrides are configured, otherwise
    return the shared default backend.

    Note: when per-session overrides are used the backend carries its own
    conversation_history.  We intentionally do NOT share it with
    default_backend so sessions are isolated.
    """
    if session.backend_url_override or session.backend_model_override:
        # Lazily create a dedicated backend for this session.  We store it
        # as an attribute so it persists across messages in the session.
        attr = "_backend_instance"
        existing = getattr(session, attr, None)
        if existing is not None:
            return existing
        backend = _create_backend(
            url=session.backend_url_override,
            model=session.backend_model_override,
        )
        # Seed with session history
        backend.conversation_history = list(session.conversation_history)
        object.__setattr__(session, attr, backend)
        return backend
    return default_backend


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Return server health status."""
    active = sum(1 for s in sessions.values() if s.connected)
    uptime = time.time() - _server_start_time if _server_start_time else 0
    return JSONResponse({
        "status": "ok",
        "active_sessions": active,
        "uptime": round(uptime, 2),
    })


# ---------------------------------------------------------------------------
# Static pages (kept from upstream)
# ---------------------------------------------------------------------------

@app.get("/")
@app.get("/voice")
@app.get("/voice/")
async def index():
    """Serve the demo page."""
    return FileResponse("src/client/index.html")


# ---------------------------------------------------------------------------
# API key management (kept from upstream)
# ---------------------------------------------------------------------------

@app.post("/api/keys")
async def create_api_key(
    name: str,
    tier: str = "free",
    master_key: Optional[str] = None,
):
    """Create a new API key (requires master key)."""
    if settings.require_auth:
        if not master_key and not settings.master_key:
            return {"error": "Master key required"}

        provided_key = master_key or ""
        if provided_key != settings.master_key:
            key = token_manager.validate_key(provided_key)
            if not key or key.tier != "enterprise":
                return {"error": "Invalid master key"}

    from .auth import PRICING_TIERS

    if tier not in PRICING_TIERS:
        return {"error": f"Invalid tier. Options: {list(PRICING_TIERS.keys())}"}

    tier_config = PRICING_TIERS[tier]
    plaintext_key, api_key = token_manager.generate_key(
        name=name,
        tier=tier,
        rate_limit=tier_config["rate_limit"],
        monthly_minutes=tier_config["monthly_minutes"],
    )
    return {
        "api_key": plaintext_key,
        "key_id": api_key.key_id,
        "name": api_key.name,
        "tier": api_key.tier,
        "monthly_minutes": api_key.monthly_minutes,
        "rate_limit": api_key.rate_limit_per_minute,
    }


@app.get("/api/usage")
async def get_usage(api_key: str):
    """Get usage stats for an API key."""
    key = token_manager.validate_key(api_key)
    if not key:
        return {"error": "Invalid API key"}
    return token_manager.get_usage(key)


# ---------------------------------------------------------------------------
# WebSocket endpoints
# ---------------------------------------------------------------------------

@app.websocket("/ws")
@app.websocket("/voice/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle voice WebSocket connections with per-session state."""

    # --- Authentication (unchanged from upstream) ---
    api_key_str = (
        websocket.query_params.get("api_key")
        or websocket.headers.get("x-api-key")
    )
    api_key: Optional[APIKey] = None

    if settings.require_auth:
        if not api_key_str:
            await websocket.close(code=4001, reason="API key required")
            return
        api_key = token_manager.validate_key(api_key_str)
        if not api_key:
            await websocket.close(code=4002, reason="Invalid API key")
            return
        if not token_manager.check_rate_limit(api_key):
            await websocket.close(code=4003, reason="Rate limit exceeded")
            return
        logger.info(f"Client connected: {api_key.name} (tier={api_key.tier})")
    else:
        if api_key_str:
            api_key = token_manager.validate_key(api_key_str)
        logger.info("Client connected (auth disabled)")

    await websocket.accept()

    # --- Session creation ---
    session_id = str(uuid.uuid4())
    session = SessionState(session_id=session_id, websocket=websocket)
    sessions[session_id] = session

    await websocket.send_json({
        "type": "session_start",
        "session_id": session_id,
    })
    logger.info(f"Session started: {session_id}")

    try:
        await _session_loop(session)
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error in session {session_id}: {e}")
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        # Mark disconnected but keep session alive for reconnection
        session.connected = False
        session.websocket = None
        session.touch()
        logger.info(
            f"Session {session_id} marked disconnected "
            f"(TTL={settings.session_ttl}s for reconnection)"
        )


# ---------------------------------------------------------------------------
# Session message loop
# ---------------------------------------------------------------------------

async def _session_loop(session: SessionState) -> None:
    """
    Main receive loop for a connected session.

    Dispatches incoming messages to the appropriate handler.
    """
    ws = session.websocket
    assert ws is not None

    while True:
        data = await ws.receive_text()
        try:
            msg = json.loads(data)
        except json.JSONDecodeError:
            logger.warning(f"Session {session.session_id}: invalid JSON received")
            await ws.send_json({"type": "error", "message": "Invalid JSON"})
            continue
        msg_type = msg.get("type", "")
        session.touch()

        if msg_type == "audio":
            await _handle_audio(session, msg)

        elif msg_type == "reconnect":
            await _handle_reconnect(session, ws, msg)

        elif msg_type == "config":
            await _handle_config(session, msg)

        elif msg_type == "start_listening":
            # Backward compatibility with upstream clients
            session.reset_vad()
            session.audio_buffer = []
            await ws.send_json({"type": "listening_started"})

        elif msg_type == "stop_listening":
            # Backward compatibility: flush any pending audio through STT
            await _flush_final_transcript(session)
            session.audio_buffer = []
            await ws.send_json({"type": "listening_stopped"})

        elif msg_type == "ping":
            await ws.send_json({"type": "pong"})

        else:
            logger.warning(
                f"Session {session.session_id}: unknown message type '{msg_type}'"
            )


# ---------------------------------------------------------------------------
# Audio handling with VAD-triggered STT
# ---------------------------------------------------------------------------

async def _handle_audio(session: SessionState, msg: dict) -> None:
    """
    Process an incoming audio chunk.

    Runs VAD continuously.  When speech ends (silence > 300 ms), the
    accumulated speech segment is flushed to STT and a partial transcript
    is sent.  After extended silence (> 1500 ms) the transcript is marked
    final and forwarded to the LLM for a response.
    """
    ws = session.websocket
    if ws is None:
        return

    raw = msg.get("data")
    if not raw:
        return

    try:
        audio_bytes = base64.b64decode(raw)
        audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
    except Exception as e:
        logger.warning(f"Session {session.session_id}: bad audio data: {e}")
        return

    if len(audio_np) == 0:
        return

    # Store in session buffer (for backward-compat stop_listening path)
    session.audio_buffer.append(audio_np)

    # --- VAD processing ---
    has_speech = True
    if vad is not None:
        has_speech = vad.is_speech(audio_np)

    # Notify client of VAD status
    await ws.send_json({
        "type": "vad_status",
        "speech_detected": has_speech,
    })

    if has_speech:
        # --- Barge-in detection ---
        if session.is_responding:
            session.interrupted = True
            session.cancel_event.set()
            await ws.send_json({"type": "interrupted"})
            logger.info(f"Session {session.session_id}: barge-in detected")
            # Wait briefly for the response task to acknowledge cancellation
            # before we start accumulating the new utterance.
            await asyncio.sleep(0.05)
            session.is_responding = False

        session.is_speech_active = True
        session.silence_samples = 0
        session.speech_segment.append(audio_np)

    else:
        # Silence
        if session.is_speech_active:
            session.silence_samples += len(audio_np)
            silence_ms = session.silence_samples / _SAMPLES_PER_MS_16K

            if silence_ms >= _SILENCE_FLUSH_MS and session.speech_segment:
                # Flush speech segment to STT (partial transcript)
                await _flush_speech_segment(session, final=False)

            if silence_ms >= _SILENCE_FINAL_MS:
                # Extended silence -- mark transcript as final
                await _flush_final_transcript(session)


async def _flush_speech_segment(session: SessionState, final: bool) -> None:
    """
    Concatenate the accumulated speech segment, send it to STT, and
    emit a transcript message.

    If *final* is False the transcript is partial; the LLM is not invoked.
    """
    ws = session.websocket
    if ws is None or not session.speech_segment:
        return

    audio_data = np.concatenate(session.speech_segment)
    session.speech_segment = []

    if len(audio_data) == 0:
        return

    try:
        transcript = await stt.transcribe(audio_data)
    except Exception as e:
        logger.error(f"Session {session.session_id}: STT error: {e}")
        transcript = ""

    if not transcript or not transcript.strip():
        return

    session.pending_transcript = (
        (session.pending_transcript + " " + transcript).strip()
    )

    await ws.send_json({
        "type": "transcript",
        "text": session.pending_transcript,
        "final": final,
    })
    logger.debug(
        f"Session {session.session_id}: "
        f"{'final' if final else 'partial'} transcript: "
        f"{session.pending_transcript}"
    )


async def _flush_final_transcript(session: SessionState) -> None:
    """
    Mark the accumulated transcript as final.

    If there is remaining audio in the speech segment, flush it first.
    Then send the transcript to the LLM and stream back the response
    with TTS.
    """
    ws = session.websocket

    # Flush any leftover speech audio
    if session.speech_segment:
        await _flush_speech_segment(session, final=True)
    elif session.pending_transcript:
        # Already flushed audio but haven't sent final yet
        if ws is not None:
            await ws.send_json({
                "type": "transcript",
                "text": session.pending_transcript,
                "final": True,
            })

    transcript = session.pending_transcript.strip()
    session.reset_vad()

    if not transcript:
        return

    logger.info(f"Session {session.session_id}: final transcript: {transcript}")

    # Send to LLM and stream TTS response
    await _generate_response(session, transcript)


# ---------------------------------------------------------------------------
# LLM response generation with TTS and barge-in cancellation
# ---------------------------------------------------------------------------

async def _generate_response(session: SessionState, user_text: str) -> None:
    """
    Stream an LLM response, synthesize TTS sentence-by-sentence, and send
    audio chunks to the client.

    Supports barge-in: if ``session.cancel_event`` is set during generation,
    we stop producing audio and text immediately.
    """
    ws = session.websocket
    if ws is None:
        return

    backend = _get_session_backend(session)

    # Prepare cancel event for this response
    session.cancel_event.clear()
    session.interrupted = False
    session.is_responding = True

    full_response = ""
    sentence_buffer = ""

    try:
        async for chunk, final_text in backend.chat_stream(
            user_text,
            history=session.conversation_history,
            cancel_event=session.cancel_event,
        ):
            # The final yield has chunk=="" and final_text=full_response
            if final_text is not None:
                full_response = final_text
                break

            full_response += chunk
            sentence_buffer += chunk

            # Send text chunk for progressive display
            if ws is not None:
                await ws.send_json({
                    "type": "response_chunk",
                    "text": chunk,
                })

            # Synthesize complete sentences
            sentence_buffer = await _synthesize_sentences(
                session, sentence_buffer
            )

        # Handle remaining text (unless interrupted)
        if not session.cancel_event.is_set() and sentence_buffer.strip():
            await _synthesize_and_send(session, sentence_buffer.strip())

        # Signal end of response
        if ws is not None:
            await ws.send_json({
                "type": "response_complete",
                "text": full_response,
            })
            logger.info(
                f"Session {session.session_id}: "
                f"response complete ({len(full_response)} chars)"
            )

    except asyncio.CancelledError:
        logger.info(f"Session {session.session_id}: response task cancelled")
    except Exception as e:
        logger.error(
            f"Session {session.session_id}: response generation error: {e}"
        )
    finally:
        session.is_responding = False

    # Update per-session conversation history with the completed exchange
    if full_response:
        session.conversation_history.append(
            {"role": "user", "content": user_text}
        )
        session.conversation_history.append(
            {"role": "assistant", "content": full_response}
        )


async def _synthesize_sentences(
    session: SessionState,
    buffer: str,
) -> str:
    """
    Extract complete sentences from *buffer*, synthesize each, and stream
    audio to the client.

    Returns the remaining (incomplete) portion of *buffer*.
    """
    separators = [". ", "! ", "? ", ".\n", "!\n", "?\n"]

    while any(sep in buffer for sep in separators):
        if session.cancel_event.is_set():
            break

        # Find the earliest sentence boundary
        earliest_idx = len(buffer)
        for sep in separators:
            idx = buffer.find(sep)
            if idx != -1 and idx < earliest_idx:
                earliest_idx = idx + len(sep)

        if earliest_idx >= len(buffer):
            break

        sentence = buffer[:earliest_idx].strip()
        buffer = buffer[earliest_idx:]

        if sentence:
            await _synthesize_and_send(session, sentence)

    return buffer


async def _synthesize_and_send(session: SessionState, text: str) -> None:
    """
    Clean *text* for speech, synthesize via TTS, and send audio chunks
    to the client.  Respects barge-in cancellation.
    """
    ws = session.websocket
    if ws is None:
        return

    speech_text = clean_for_speech(text)
    if not speech_text:
        return

    logger.debug(
        f"Session {session.session_id}: synthesizing: {speech_text[:60]}..."
    )

    try:
        async for audio_chunk in tts.synthesize_stream(speech_text):
            if session.cancel_event.is_set():
                break
            audio_b64 = base64.b64encode(audio_chunk).decode()
            await ws.send_json({
                "type": "audio_chunk",
                "data": audio_b64,
                "sample_rate": 24000,
            })
    except Exception as e:
        logger.error(f"Session {session.session_id}: TTS error: {e}")


# ---------------------------------------------------------------------------
# Reconnection
# ---------------------------------------------------------------------------

async def _handle_reconnect(
    session: SessionState,
    ws: WebSocket,
    msg: dict,
) -> None:
    """
    Handle a reconnect request.

    The client sends ``{"type":"reconnect","session_id":"..."}`` to resume
    a previous session.  If the old session exists and has not expired, we
    migrate its state into the current session and notify the client.
    """
    old_id = msg.get("session_id")
    if not old_id or old_id not in sessions:
        await ws.send_json({
            "type": "error",
            "message": "Session not found or expired",
        })
        return

    old_session = sessions[old_id]
    if old_session.is_expired(settings.session_ttl):
        del sessions[old_id]
        await ws.send_json({
            "type": "error",
            "message": "Session expired",
        })
        return

    # Migrate conversation history and config overrides to the current session
    session.conversation_history = list(old_session.conversation_history)
    session.backend_url_override = old_session.backend_url_override
    session.backend_model_override = old_session.backend_model_override

    # Transfer any cached backend instance
    old_backend = getattr(old_session, "_backend_instance", None)
    if old_backend is not None:
        object.__setattr__(session, "_backend_instance", old_backend)

    # Re-key: remove old session, re-register current session under the old ID
    # so the client keeps using its original session_id.
    new_id = session.session_id
    del sessions[old_id]
    session.session_id = old_id
    sessions[old_id] = session
    if new_id in sessions and new_id != old_id:
        del sessions[new_id]

    await ws.send_json({
        "type": "session_start",
        "session_id": old_id,
        "resumed": True,
        "history_length": len(session.conversation_history),
    })
    logger.info(
        f"Session {new_id} reconnected as {old_id} "
        f"({len(session.conversation_history)} history entries)"
    )


# ---------------------------------------------------------------------------
# Per-session config override
# ---------------------------------------------------------------------------

async def _handle_config(session: SessionState, msg: dict) -> None:
    """
    Handle a client config message.

    Example payload::

        {
            "type": "config",
            "agent": {
                "backend_url": "https://my-custom-llm/v1",
                "model": "my-model"
            }
        }

    This allows clients to redirect the LLM backend on a per-session basis.
    """
    ws = session.websocket
    agent = msg.get("agent", {})
    if not agent:
        return

    if "backend_url" in agent:
        session.backend_url_override = agent["backend_url"]
    if "model" in agent:
        session.backend_model_override = agent["model"]

    # Invalidate any cached backend instance so it gets recreated
    if hasattr(session, "_backend_instance"):
        try:
            delattr(session, "_backend_instance")
        except AttributeError:
            pass

    if ws is not None:
        await ws.send_json({
            "type": "config_ack",
            "backend_url": session.backend_url_override,
            "model": session.backend_model_override,
        })
    logger.info(
        f"Session {session.session_id}: config updated "
        f"(url={session.backend_url_override}, model={session.backend_model_override})"
    )


# ---------------------------------------------------------------------------
# Session cleanup
# ---------------------------------------------------------------------------

async def _session_cleanup_loop() -> None:
    """
    Background task that periodically removes expired sessions.

    Runs every 60 seconds.  A session is expired when it has been
    disconnected for longer than ``settings.session_ttl`` seconds.
    """
    try:
        while True:
            await asyncio.sleep(60)
            now = time.time()
            expired_ids = [
                sid
                for sid, s in sessions.items()
                if not s.connected and s.is_expired(settings.session_ttl)
            ]
            for sid in expired_ids:
                del sessions[sid]
                logger.debug(f"Cleaned up expired session: {sid}")
            if expired_ids:
                logger.info(
                    f"Session cleanup: removed {len(expired_ids)} expired session(s), "
                    f"{len(sessions)} remaining"
                )
    except asyncio.CancelledError:
        pass


# ---------------------------------------------------------------------------
# Static files (kept from upstream)
# ---------------------------------------------------------------------------

client_dir = Path(__file__).parent.parent / "client"
if client_dir.exists():
    app.mount("/static", StaticFiles(directory=str(client_dir)), name="static")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.server.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
