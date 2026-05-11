"""
Voice Relay Server -- Phase 2

Builds on Phase 1 with:
  - SQLite persistence via async database layer (sessions, agents, keys, usage)
  - Multi-agent routing via AgentRouter (DB-backed + env-var fallback)
  - Admin REST API mounted at /admin (master-key protected)
  - DB-backed API key validation replacing in-memory token_manager
  - Usage tracking (STT seconds, TTS seconds, LLM tokens, session events)
  - Monthly minute quota enforcement per API key
  - CORS middleware for widget embedding
  - Session persistence across server restarts (DB-backed recovery)

Backward compatibility:
  - All Phase 1 env vars still honoured
  - OPENCLAW_REQUIRE_AUTH=false disables auth (dev mode)
  - Demo page at / and /voice still works
  - WebSocket protocol at /ws and /voice/ws unchanged
  - Per-session config override via "config" message type
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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic_settings import BaseSettings

from .stt import WhisperSTT
from .tts import RelayTTS
from .backend import AIBackend
from .vad import VoiceActivityDetector
from .database import Database
from .agents import AgentRouter
from .admin import admin_router, init_admin
from .text_utils import clean_for_speech


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Server configuration covering Phase 1 and Phase 2 settings."""

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

    # AI Backend (kept for backward compatibility / env-var fallback)
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

    # --- Phase 2 additions ---

    # SQLite database path
    db_path: Optional[str] = None

    # CORS origins (comma-separated, or "*" for all)
    cors_origins: str = "*"

    class Config:
        env_prefix = "OPENCLAW_"
        env_file = ".env"


def _load_extra_env(settings: "Settings") -> "Settings":
    """
    Load environment variables that use non-OPENCLAW_ prefixes.

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


def _resolve_db_path(settings: "Settings") -> str:
    """
    Determine the SQLite database file path.

    Priority:
      1. OPENCLAW_DB_PATH env var / settings.db_path
      2. $OPENCLAW_STATE_DIR/voice_relay.db
      3. ./voice_relay.db
    """
    if settings.db_path:
        return settings.db_path

    env_db = os.getenv("OPENCLAW_DB_PATH")
    if env_db:
        return env_db

    state_dir = os.getenv("OPENCLAW_STATE_DIR")
    if state_dir:
        os.makedirs(state_dir, exist_ok=True)
        return os.path.join(state_dir, "voice_relay.db")

    return "./voice_relay.db"


settings = Settings()
settings = _load_extra_env(settings)


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

    # --- Phase 2 additions ---
    api_key_id: Optional[str] = None        # Which API key owns this session
    api_key_record: Optional[dict] = None    # Cached key metadata
    agent_id: Optional[str] = None           # Which agent backend is assigned

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

app = FastAPI(title="Voice Relay", version="2.0.0-phase2")

# CORS middleware for widget embedding
_cors_origins = [
    o.strip()
    for o in settings.cors_origins.split(",")
    if o.strip()
] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount admin API
app.include_router(admin_router, prefix="/admin")

# Global instances (initialized on startup)
stt: Optional[WhisperSTT] = None
tts: Optional[RelayTTS] = None
vad: Optional[VoiceActivityDetector] = None
db: Optional[Database] = None
agent_router: Optional[AgentRouter] = None

# Session registry (in-memory; authoritative for live WebSocket state)
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
    """Initialize all subsystems on server start."""
    global stt, tts, vad, db, agent_router, _server_start_time, _cleanup_task

    _server_start_time = time.time()
    logger.info("Initializing Voice Relay server (Phase 2)...")

    # --- Database ---
    db_path = _resolve_db_path(settings)
    logger.info(f"Database path: {db_path}")
    db = Database(db_path)
    await db.initialize()

    # --- Agent router (seeds default agent from env if DB is empty) ---
    agent_router = AgentRouter(db)
    await agent_router.seed_defaults()

    # --- Auth mode ---
    if settings.require_auth:
        logger.info("Authentication ENABLED (DB-backed API keys)")
    else:
        logger.warning("Authentication DISABLED (dev mode)")

    # --- STT ---
    logger.info(f"Loading STT model: {settings.stt_model}")
    stt = WhisperSTT(
        model_name=settings.stt_model,
        device=settings.stt_device,
        api_url=os.environ.get("VSAAS_URL"),
        api_key=os.environ.get("VSAAS_API_KEY"),
    )

    # --- TTS ---
    logger.info(f"Loading TTS model: {settings.tts_model}")
    tts = RelayTTS(
        voice=settings.tts_voice,
    )

    # --- VAD ---
    logger.info("Loading VAD model")
    vad = VoiceActivityDetector()

    # --- Admin API init ---
    # Set backend identifiers for the admin stats endpoint
    os.environ["_ADMIN_STT_BACKEND"] = getattr(stt, "backend", "unknown")
    os.environ["_ADMIN_TTS_BACKEND"] = getattr(tts, "backend", "unknown")
    os.environ["_ADMIN_VAD_BACKEND"] = getattr(vad, "backend", "unknown")
    init_admin(db, sessions, _server_start_time)

    # --- Session cleanup loop ---
    _cleanup_task = asyncio.create_task(_session_cleanup_loop())

    logger.info("Voice Relay server ready")


@app.on_event("shutdown")
async def shutdown():
    """Gracefully shut down background tasks and the database."""
    global db

    # Cancel cleanup task
    if _cleanup_task and not _cleanup_task.done():
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            pass

    # Mark all in-memory sessions as disconnected in DB
    if db is not None:
        for sid, session in sessions.items():
            try:
                await db.update_session(sid, is_connected=False)
            except Exception as exc:
                logger.warning(f"Failed to mark session {sid} disconnected on shutdown: {exc}")

        await db.close()
        db = None

    logger.info("Voice Relay server shut down")


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Return server health status including database connectivity."""
    active = sum(1 for s in sessions.values() if s.connected)
    uptime = time.time() - _server_start_time if _server_start_time else 0
    db_connected = db is not None and db._conn is not None
    return JSONResponse({
        "status": "ok",
        "active_sessions": active,
        "uptime": round(uptime, 2),
        "db_connected": db_connected,
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


@app.get("/agents")
async def public_agents():
    """
    Return a lightweight list of available agents for the voice client UI.

    This is a PUBLIC endpoint (no admin auth required) that returns only
    the agent id, name, and default flag — no API keys, URLs, or other
    sensitive config.
    """
    if db is None:
        return JSONResponse([])
    try:
        records = await db.list_agents()
        return JSONResponse([
            {
                "id": r["id"],
                "name": r["name"],
                "is_default": bool(r.get("is_default", False)),
            }
            for r in records
        ])
    except Exception as exc:
        logger.warning(f"Failed to fetch public agent list: {exc}")
        return JSONResponse([])


# ---------------------------------------------------------------------------
# Usage tracking helpers
# ---------------------------------------------------------------------------

async def _record_usage(
    api_key_id: Optional[str],
    event_type: str,
    value: float,
    session_id: Optional[str] = None,
) -> None:
    """
    Record a usage event in the database.

    Silently skips recording if authentication is disabled (no api_key_id)
    or if the database is unavailable.  This ensures usage tracking never
    blocks the voice pipeline.
    """
    if not api_key_id or db is None:
        return
    try:
        await db.record_usage(
            api_key_id=api_key_id,
            event_type=event_type,
            value=value,
            session_id=session_id,
        )
    except Exception as exc:
        logger.warning(f"Failed to record usage ({event_type}): {exc}")


def _estimate_audio_seconds(audio_np: np.ndarray, sample_rate: int = 16000) -> float:
    """Estimate audio duration in seconds from a numpy array at the given sample rate."""
    if len(audio_np) == 0:
        return 0.0
    return len(audio_np) / sample_rate


def _estimate_tts_seconds(text: str) -> float:
    """
    Estimate TTS output duration from text length.

    Rough heuristic: average speaking rate is about 150 words per minute,
    or ~2.5 words per second.  Average word length ~5 chars, so ~12.5
    chars per second.
    """
    if not text:
        return 0.0
    return max(len(text) / 12.5, 0.1)


def _estimate_llm_tokens(text: str) -> int:
    """
    Estimate the number of LLM tokens from response text.

    Rough heuristic: ~4 characters per token for English text.
    """
    if not text:
        return 0
    return max(len(text) // 4, 1)


# ---------------------------------------------------------------------------
# WebSocket endpoints
# ---------------------------------------------------------------------------

@app.websocket("/ws")
@app.websocket("/voice/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle voice WebSocket connections with per-session state and DB persistence."""

    # --- Authentication ---
    api_key_str = (
        websocket.query_params.get("api_key")
        or websocket.headers.get("x-api-key")
    )
    api_key_record: Optional[dict] = None

    if settings.require_auth:
        if not api_key_str:
            await websocket.close(code=4001, reason="API key required")
            return

        # Validate against the database
        api_key_record = await db.validate_api_key(api_key_str) if db else None
        if not api_key_record:
            await websocket.close(code=4002, reason="Invalid API key")
            return

        # Check monthly minute limit (0 = unlimited)
        monthly_limit = api_key_record.get("monthly_minutes", 0)
        if monthly_limit and monthly_limit > 0:
            minutes_used = await db.get_monthly_minutes(api_key_record["id"])
            if minutes_used >= monthly_limit:
                await websocket.close(
                    code=4004,
                    reason="Monthly minute quota exceeded",
                )
                return

        logger.info(
            f"Client connected: {api_key_record.get('name', 'unknown')} "
            f"(tier={api_key_record.get('tier', 'unknown')})"
        )
    else:
        # Auth disabled -- optionally validate a key if provided
        if api_key_str and db:
            api_key_record = await db.validate_api_key(api_key_str)
        logger.info("Client connected (auth disabled)")

    await websocket.accept()

    # --- Session creation ---
    session_id = str(uuid.uuid4())
    session = SessionState(
        session_id=session_id,
        websocket=websocket,
        api_key_id=api_key_record["id"] if api_key_record else None,
        api_key_record=api_key_record,
        agent_id=api_key_record.get("agent_id") if api_key_record else None,
    )
    sessions[session_id] = session

    # Persist session to DB
    if db is not None:
        try:
            await db.create_session(
                session_id=session_id,
                api_key_id=session.api_key_id,
                agent_id=session.agent_id,
            )
        except Exception as exc:
            logger.warning(f"Failed to persist session {session_id} to DB: {exc}")

    # Record session_start usage event
    await _record_usage(
        api_key_id=session.api_key_id,
        event_type="session_start",
        value=0,
        session_id=session_id,
    )

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

        # Update DB
        if db is not None:
            try:
                await db.update_session(session_id, is_connected=False)
            except Exception as exc:
                logger.warning(
                    f"Failed to update session {session_id} disconnect in DB: {exc}"
                )

        # Record session_end usage event
        await _record_usage(
            api_key_id=session.api_key_id,
            event_type="session_end",
            value=0,
            session_id=session_id,
        )

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
    Records STT usage based on audio duration.
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

    # Record STT usage (estimate seconds from audio length)
    stt_seconds = _estimate_audio_seconds(audio_data, settings.sample_rate)
    await _record_usage(
        api_key_id=session.api_key_id,
        event_type="stt_seconds",
        value=stt_seconds,
        session_id=session.session_id,
    )

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

    Uses the AgentRouter to resolve the correct backend for this session.
    Records LLM and TTS usage after each operation.
    """
    ws = session.websocket
    if ws is None:
        return

    # Resolve the AI backend for this session via AgentRouter
    backend = await _get_session_backend(session)

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

    # Record LLM usage (estimate tokens from response length)
    if full_response:
        estimated_tokens = _estimate_llm_tokens(user_text + full_response)
        await _record_usage(
            api_key_id=session.api_key_id,
            event_type="llm_tokens",
            value=estimated_tokens,
            session_id=session.session_id,
        )

    # Update per-session conversation history with the completed exchange
    if full_response:
        session.conversation_history.append(
            {"role": "user", "content": user_text}
        )
        session.conversation_history.append(
            {"role": "assistant", "content": full_response}
        )

        # Persist conversation history to DB
        if db is not None:
            try:
                await db.save_conversation_history(
                    session.session_id,
                    session.conversation_history,
                )
            except Exception as exc:
                logger.warning(
                    f"Failed to persist conversation history for "
                    f"session {session.session_id}: {exc}"
                )


async def _get_session_backend(session: SessionState) -> AIBackend:
    """
    Resolve the AI backend for a session using the AgentRouter.

    Falls back to constructing a backend from env vars if the router
    is not yet initialised (should not happen in normal operation).
    """
    if agent_router is not None:
        return await agent_router.resolve_backend(
            session,
            api_key_record=session.api_key_record,
        )

    # Fallback: construct from env vars directly (Phase 1 compat)
    logger.warning(
        f"Session {session.session_id}: AgentRouter not available, "
        "using env-var backend fallback"
    )
    return _create_fallback_backend(
        url=session.backend_url_override,
        model=session.backend_model_override,
    )


def _create_fallback_backend(
    url: Optional[str] = None,
    model: Optional[str] = None,
) -> AIBackend:
    """
    Create an AIBackend instance directly from env vars.

    This is the Phase 1 fallback path, used only when the AgentRouter
    is not available (e.g. during early startup or in tests).
    """
    gateway_url = settings.openclaw_gateway_url or os.getenv("OPENCLAW_GATEWAY_URL")
    gateway_token = settings.openclaw_gateway_token or os.getenv("OPENCLAW_GATEWAY_TOKEN")

    if gateway_url and gateway_token:
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

    Records TTS usage based on estimated audio duration.
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

    # Record TTS usage (estimate seconds from text length)
    tts_seconds = _estimate_tts_seconds(speech_text)
    await _record_usage(
        api_key_id=session.api_key_id,
        event_type="tts_seconds",
        value=tts_seconds,
        session_id=session.session_id,
    )


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
    a previous session.  The handler checks both the in-memory registry and
    the database, enabling recovery across server restarts.
    """
    old_id = msg.get("session_id")
    if not old_id:
        await ws.send_json({
            "type": "error",
            "message": "Session ID required for reconnection",
        })
        return

    old_session = sessions.get(old_id)

    # --- Attempt DB-based recovery if not in memory ---
    if old_session is None and db is not None:
        try:
            db_record = await db.get_session(old_id)
            if db_record is not None:
                logger.info(
                    f"Recovering session {old_id} from database "
                    f"({len(db_record.get('conversation_history', []))} history entries)"
                )
                # Rebuild a SessionState from the DB record
                old_session = SessionState(
                    session_id=old_id,
                    conversation_history=db_record.get("conversation_history", []),
                    backend_url_override=db_record.get("backend_url_override"),
                    backend_model_override=db_record.get("backend_model_override"),
                    connected=False,
                    api_key_id=db_record.get("api_key_id"),
                    agent_id=db_record.get("agent_id"),
                )
                # Re-populate api_key_record if we have a key ID
                if old_session.api_key_id and db is not None:
                    keys = await db.list_api_keys()
                    old_session.api_key_record = next(
                        (k for k in keys if k["id"] == old_session.api_key_id),
                        None,
                    )
                sessions[old_id] = old_session
        except Exception as exc:
            logger.warning(f"DB session recovery failed for {old_id}: {exc}")

    if old_session is None:
        await ws.send_json({
            "type": "error",
            "message": "Session not found or expired",
        })
        return

    if old_session.is_expired(settings.session_ttl):
        # Clean up the expired session
        sessions.pop(old_id, None)
        await ws.send_json({
            "type": "error",
            "message": "Session expired",
        })
        return

    # Migrate conversation history and config overrides to the current session
    session.conversation_history = list(old_session.conversation_history)
    session.backend_url_override = old_session.backend_url_override
    session.backend_model_override = old_session.backend_model_override
    session.api_key_id = old_session.api_key_id
    session.api_key_record = old_session.api_key_record
    session.agent_id = old_session.agent_id

    # Transfer any cached backend instance
    old_backend = getattr(old_session, "_backend_instance", None)
    if old_backend is not None:
        object.__setattr__(session, "_backend_instance", old_backend)

    # Re-key: remove old session, re-register current session under the old ID
    new_id = session.session_id
    sessions.pop(old_id, None)
    session.session_id = old_id
    sessions[old_id] = session
    if new_id in sessions and new_id != old_id:
        del sessions[new_id]

    # Update DB: mark session as connected again
    if db is not None:
        try:
            await db.update_session(old_id, is_connected=True)
        except Exception as exc:
            logger.warning(f"Failed to update reconnected session {old_id} in DB: {exc}")

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

    Supports two modes:

    1. **Agent selection** (Phase 2 preferred) — pick a registered agent by ID::

        {"type": "config", "agent_id": "openclaw"}

    2. **Raw override** (Phase 1 compat) — specify backend URL and model directly::

        {"type": "config", "agent": {"backend_url": "https://...", "model": "..."}}

    If ``agent_id`` is provided, it takes priority: the session's
    ``backend_url_override`` and ``backend_model_override`` are set from the
    looked-up agent record so the AgentRouter resolves correctly.
    """
    ws = session.websocket

    # --- Mode 1: agent_id selection (Phase 2) ---
    agent_id = msg.get("agent_id")
    if agent_id and db is not None:
        try:
            agent_record = await db.get_agent(agent_id)
            if agent_record:
                # Set agent_id on session -- AgentRouter resolves the full
                # agent config (url, model, system_prompt, api_key) from the
                # DB via get_or_create_backend.  We intentionally do NOT set
                # backend_url_override / backend_model_override here because
                # that would bypass the cached-backend path and lose the
                # agent's system_prompt.
                session.agent_id = agent_id
                session.backend_url_override = None
                session.backend_model_override = None

                # Invalidate any cached per-session backend so the router
                # re-resolves on the next voice turn
                if hasattr(session, "_backend_instance"):
                    try:
                        delattr(session, "_backend_instance")
                    except AttributeError:
                        pass

                # Also invalidate the AgentRouter cache for this agent so
                # any recent admin changes are picked up
                if agent_router is not None:
                    agent_router.invalidate_cache(agent_id)

                # Persist to DB
                try:
                    await db.update_session(
                        session.session_id,
                        agent_id=agent_id,
                    )
                except Exception as exc:
                    logger.warning(
                        f"Failed to persist agent config for session "
                        f"{session.session_id}: {exc}"
                    )

                if ws is not None:
                    await ws.send_json({
                        "type": "config_ack",
                        "agent_id": agent_id,
                        "agent_name": agent_record.get("name", agent_id),
                        "model": agent_record.get("model"),
                    })
                logger.info(
                    f"Session {session.session_id}: switched to agent "
                    f"{agent_id!r} (model={agent_record.get('model')})"
                )
                return
            else:
                logger.warning(f"Agent {agent_id!r} not found in DB")
                if ws is not None:
                    await ws.send_json({
                        "type": "error",
                        "message": f"Agent '{agent_id}' not found",
                    })
                return
        except Exception as exc:
            logger.error(f"Error looking up agent {agent_id!r}: {exc}")

    # --- Mode 2: raw override (Phase 1 compat) ---
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

    # Persist the overrides to DB
    if db is not None:
        try:
            await db.update_session(
                session.session_id,
                backend_url_override=session.backend_url_override,
                backend_model_override=session.backend_model_override,
            )
        except Exception as exc:
            logger.warning(
                f"Failed to persist config overrides for session "
                f"{session.session_id}: {exc}"
            )

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

    Runs every 60 seconds.  Cleans up both the in-memory registry and the
    database.  Records session_end usage events for cleaned-up sessions.
    """
    try:
        while True:
            await asyncio.sleep(60)

            # --- In-memory cleanup ---
            expired_ids = [
                sid
                for sid, s in sessions.items()
                if not s.connected and s.is_expired(settings.session_ttl)
            ]
            for sid in expired_ids:
                expired_session = sessions.pop(sid, None)

                # Record session_end for expired sessions that had a key
                if expired_session and expired_session.api_key_id:
                    await _record_usage(
                        api_key_id=expired_session.api_key_id,
                        event_type="session_end",
                        value=0,
                        session_id=sid,
                    )

                logger.debug(f"Cleaned up expired session: {sid}")

            if expired_ids:
                logger.info(
                    f"Session cleanup: removed {len(expired_ids)} expired session(s) "
                    f"from memory, {len(sessions)} remaining"
                )

            # --- DB cleanup ---
            if db is not None:
                try:
                    db_deleted = await db.delete_expired_sessions(settings.session_ttl)
                    if db_deleted:
                        logger.info(
                            f"Session cleanup: removed {db_deleted} expired session(s) "
                            "from database"
                        )
                except Exception as exc:
                    logger.warning(f"DB session cleanup failed: {exc}")

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
