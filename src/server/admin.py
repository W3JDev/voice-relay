"""
Admin REST API for Voice Relay Phase 2.

All endpoints require the master key via ``Authorization: Bearer <key>``
or the ``X-Admin-Key`` header.  The master key is read from the
``OPENCLAW_MASTER_KEY`` or ``SETUP_PASSWORD`` environment variable.

Mount on the main app with::

    from .admin import admin_router, init_admin
    init_admin(db)
    app.include_router(admin_router, prefix="/admin")

Endpoint summary
----------------
Agents
  POST   /admin/agents              Create a new agent backend
  GET    /admin/agents              List all agents
  GET    /admin/agents/{agent_id}   Get agent details
  PUT    /admin/agents/{agent_id}   Update an agent
  DELETE /admin/agents/{agent_id}   Delete an agent

API Keys
  POST   /admin/keys                Create a new API key (plaintext returned once)
  GET    /admin/keys                List all keys (prefix only, no plaintext)
  GET    /admin/keys/{key_id}       Key details + usage summary
  DELETE /admin/keys/{key_id}       Deactivate a key

Sessions
  GET    /admin/sessions            List active sessions
  GET    /admin/sessions/{sid}      Session details
  DELETE /admin/sessions/{sid}      Force-close a session

Usage
  GET    /admin/usage/{key_id}            Usage summary for a key
  GET    /admin/usage/{key_id}/events     Raw usage events (paginated)

System
  GET    /admin/stats               System-wide stats
"""

from __future__ import annotations

import os
import secrets
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field

from .database import Database
from .models import (
    AgentConfig,
    APIKeyCreate,
    APIKeyResponse,
    SessionInfo,
    SystemStats,
    UsageSummary,
)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

admin_router = APIRouter(tags=["admin"])

#: Database instance injected at startup via :func:`init_admin`.
_db: Optional[Database] = None

#: Reference to the in-memory session registry kept in main.py.
#: Injected at startup via :func:`init_admin` so the admin API can list /
#: force-close live sessions without coupling tightly to main.py.
_sessions: Optional[Dict[str, Any]] = None

#: Server start time (epoch seconds) for uptime reporting.
_server_start: float = 0.0

#: Reference to the AgentRouter for cache invalidation on admin changes.
_agent_router: Optional[Any] = None


def init_admin(
    db: Database,
    sessions: Optional[Dict[str, Any]] = None,
    server_start: float = 0.0,
    agent_router: Optional[Any] = None,
) -> None:
    """Inject dependencies from the main application at startup.

    Call this once from the FastAPI ``startup`` event handler **before**
    any requests arrive.

    Args:
        db:           Initialised :class:`~database.Database` instance.
        sessions:     The ``sessions`` dict from ``main.py`` (maps session
                      ID → :class:`~main.SessionState`).  Pass ``None`` if
                      the main module is not yet wired up; the admin session
                      endpoints will return empty results.
        server_start: ``time.time()`` value captured when the server started,
                      used to compute the ``uptime`` field in
                      ``GET /admin/stats``.
        agent_router: The :class:`~agents.AgentRouter` instance for cache
                      invalidation when agents are updated or deleted.
    """
    global _db, _sessions, _server_start, _agent_router
    _db = db
    _sessions = sessions
    _server_start = server_start or time.time()
    _agent_router = agent_router
    logger.info("Admin API initialised")


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

async def require_admin(
    authorization: Optional[str] = Header(None),
    x_admin_key: Optional[str] = Header(None),
) -> None:
    """FastAPI dependency that enforces master-key authentication.

    Accepts the key as:
    - ``Authorization: Bearer <master_key>``
    - ``X-Admin-Key: <master_key>``

    Raises:
        HTTPException 503: Admin API is not configured (no master key in env).
        HTTPException 401: The supplied token is missing or incorrect.
    """
    master_key = os.environ.get("OPENCLAW_MASTER_KEY") or os.environ.get("SETUP_PASSWORD")
    if not master_key:
        raise HTTPException(
            status_code=503,
            detail="Admin API not configured (set OPENCLAW_MASTER_KEY)",
        )

    token: Optional[str] = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
    elif x_admin_key:
        token = x_admin_key

    if not token or not secrets.compare_digest(token, master_key):
        raise HTTPException(status_code=401, detail="Invalid admin key")


def _require_db() -> Database:
    """Return the injected database or raise 503 if not yet initialised."""
    if _db is None:
        raise HTTPException(
            status_code=503,
            detail="Admin API database not initialised; call init_admin() at startup",
        )
    return _db


# ---------------------------------------------------------------------------
# Request / response models (admin-specific, not in models.py)
# ---------------------------------------------------------------------------

class AgentUpdate(BaseModel):
    """Optional fields for updating an existing agent."""

    name: Optional[str] = Field(None, description="Human-readable display name.")
    backend_type: Optional[str] = Field(None, description="'openai' or 'openclaw'.")
    url: Optional[str] = Field(None, description="Base URL of the LLM endpoint.")
    model: Optional[str] = Field(None, description="Model identifier.")
    api_key: Optional[str] = Field(None, description="API key for the backend.")
    system_prompt: Optional[str] = Field(None, description="System prompt override.")
    voice: Optional[str] = Field(None, description="Kokoro TTS voice ID (e.g. 'am_adam', 'af_bella').")
    is_default: Optional[bool] = Field(None, description="Set as system-wide default.")


class APIKeyCreateResponse(BaseModel):
    """Response returned when a new API key is created.

    The ``plaintext_key`` field is populated **only** in the create response
    and is never stored or shown again.
    """

    plaintext_key: str = Field(
        ...,
        description="The full API key — store it now, it cannot be retrieved later.",
    )
    key: APIKeyResponse = Field(..., description="Metadata record for the new key.")


class UsageEvent(BaseModel):
    """A single raw usage event row."""

    id: int
    api_key_id: str
    session_id: Optional[str]
    event_type: str
    value: float
    created_at: datetime


class UsageEventsPage(BaseModel):
    """Paginated list of raw usage events."""

    key_id: str
    total: int = Field(..., description="Total matching events (before pagination).")
    offset: int
    limit: int
    events: List[UsageEvent]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _to_api_key_response(record: dict) -> APIKeyResponse:
    """Convert a raw DB dict to :class:`~models.APIKeyResponse`."""
    return APIKeyResponse(
        id=record["id"],
        key_prefix=record["key_prefix"],
        name=record["name"],
        tier=record["tier"],
        rate_limit_per_minute=record["rate_limit_per_minute"],
        monthly_minutes=record["monthly_minutes"],
        is_active=bool(record["is_active"]),
        agent_id=record.get("agent_id"),
        created_at=_parse_dt(record["created_at"]),
        expires_at=_parse_dt(record["expires_at"]) if record.get("expires_at") else None,
    )


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO 8601 string (possibly with or without TZ) to a datetime."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _to_session_info(record: dict) -> SessionInfo:
    """Convert a raw DB session dict to :class:`~models.SessionInfo`."""
    history = record.get("conversation_history") or []
    if isinstance(history, str):
        import json
        history = json.loads(history)

    return SessionInfo(
        id=record["id"],
        api_key_id=record.get("api_key_id"),
        agent_id=record.get("agent_id"),
        is_connected=bool(record.get("is_connected", False)),
        history_length=len(history),
        created_at=_parse_dt(record["created_at"]) or datetime.now(timezone.utc),
        last_activity=_parse_dt(record["last_activity"]) or datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Agent endpoints
# ---------------------------------------------------------------------------

@admin_router.post(
    "/agents",
    response_model=AgentConfig,
    status_code=201,
    summary="Create a new agent backend",
)
async def create_agent(
    body: AgentConfig,
    _: None = Depends(require_admin),
    db: Database = Depends(_require_db),
) -> AgentConfig:
    """Register a new AI backend agent.

    The ``id`` field must be a unique slug (e.g. ``"hermes"``).  If
    ``is_default`` is ``True``, any existing default agent is unset first —
    there is always at most one system-wide default.

    Returns the newly created agent record.
    """
    existing = await db.get_agent(body.id)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Agent with id '{body.id}' already exists",
        )

    record = await db.create_agent(
        id=body.id,
        name=body.name,
        url=body.url,
        model=body.model,
        backend_type=body.backend_type,
        api_key=body.api_key,
        system_prompt=body.system_prompt,
        is_default=body.is_default,
        voice=body.voice,
    )

    logger.info(f"Admin: created agent {body.id!r}")
    return AgentConfig(**record)


@admin_router.get(
    "/agents",
    response_model=List[AgentConfig],
    summary="List all agent backends",
)
async def list_agents(
    _: None = Depends(require_admin),
    db: Database = Depends(_require_db),
) -> List[AgentConfig]:
    """Return all registered agent configurations.

    The system default agent (if any) is listed first.
    """
    records = await db.list_agents()
    return [AgentConfig(**r) for r in records]


@admin_router.get(
    "/agents/{agent_id}",
    response_model=AgentConfig,
    summary="Get agent details",
)
async def get_agent(
    agent_id: str,
    _: None = Depends(require_admin),
    db: Database = Depends(_require_db),
) -> AgentConfig:
    """Fetch a single agent configuration by its slug.

    Raises **404** if the agent does not exist.
    """
    record = await db.get_agent(agent_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id!r}")
    return AgentConfig(**record)


@admin_router.put(
    "/agents/{agent_id}",
    response_model=AgentConfig,
    summary="Update an agent backend",
)
async def update_agent(
    agent_id: str,
    body: AgentUpdate,
    _: None = Depends(require_admin),
    db: Database = Depends(_require_db),
) -> AgentConfig:
    """Update one or more fields on an existing agent.

    Only the fields present in the request body are changed; omitted fields
    retain their current values.

    Raises **404** if the agent does not exist.
    Raises **422** if the request body is empty (no fields to update).
    """
    record = await db.get_agent(agent_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id!r}")

    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=422, detail="No fields provided to update")

    await db.update_agent(agent_id, **updates)

    # Invalidate cached backend so next session picks up updated config
    if _agent_router is not None:
        _agent_router.invalidate_cache(agent_id)
        logger.debug(f"Admin: invalidated backend cache for agent {agent_id!r}")

    updated = await db.get_agent(agent_id)
    assert updated is not None

    logger.info(f"Admin: updated agent {agent_id!r} fields={list(updates)}")
    return AgentConfig(**updated)


@admin_router.delete(
    "/agents/{agent_id}",
    status_code=204,
    summary="Delete an agent backend",
)
async def delete_agent(
    agent_id: str,
    _: None = Depends(require_admin),
    db: Database = Depends(_require_db),
) -> None:
    """Permanently remove an agent configuration.

    Sessions and API keys that reference the deleted agent will have their
    ``agent_id`` set to ``NULL`` (SQLite foreign-key cascade).

    Raises **404** if the agent does not exist.
    """
    record = await db.get_agent(agent_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id!r}")

    await db.delete_agent(agent_id)

    # Invalidate cached backend for the deleted agent
    if _agent_router is not None:
        _agent_router.invalidate_cache(agent_id)
        logger.debug(f"Admin: invalidated backend cache for deleted agent {agent_id!r}")

    logger.info(f"Admin: deleted agent {agent_id!r}")


# ---------------------------------------------------------------------------
# API Key endpoints
# ---------------------------------------------------------------------------

@admin_router.post(
    "/keys",
    response_model=APIKeyCreateResponse,
    status_code=201,
    summary="Create a new API key",
)
async def create_api_key(
    body: APIKeyCreate,
    _: None = Depends(require_admin),
    db: Database = Depends(_require_db),
) -> APIKeyCreateResponse:
    """Generate a new API key and return the plaintext key **once**.

    The plaintext key is **never stored** in the database (only its SHA-256
    hash is kept).  Record the returned ``plaintext_key`` securely — it
    cannot be retrieved again.

    If ``agent_id`` is specified, the referenced agent must already exist.
    """
    # Validate agent_id if provided
    if body.agent_id:
        agent = await db.get_agent(body.agent_id)
        if agent is None:
            raise HTTPException(
                status_code=404,
                detail=f"Agent not found: {body.agent_id!r}",
            )

    expires_at_str: Optional[str] = None
    if body.expires_at:
        expires_at_str = body.expires_at.isoformat()

    plaintext_key, record = await db.create_api_key(
        name=body.name,
        tier=body.tier,
        agent_id=body.agent_id,
        rate_limit_per_minute=body.rate_limit_per_minute,
        monthly_minutes=body.monthly_minutes,
        expires_at=expires_at_str,
    )

    logger.info(f"Admin: created API key {record['id']!r} name={body.name!r}")
    return APIKeyCreateResponse(
        plaintext_key=plaintext_key,
        key=_to_api_key_response(record),
    )


@admin_router.get(
    "/keys",
    response_model=List[APIKeyResponse],
    summary="List all API keys",
)
async def list_api_keys(
    _: None = Depends(require_admin),
    db: Database = Depends(_require_db),
) -> List[APIKeyResponse]:
    """Return all API key records, most recently created first.

    Plaintext keys are never included — use ``key_prefix`` for identification.
    """
    records = await db.list_api_keys()
    return [_to_api_key_response(r) for r in records]


@admin_router.get(
    "/keys/{key_id}",
    response_model=Dict[str, Any],
    summary="Get key details and current-month usage summary",
)
async def get_api_key(
    key_id: str,
    _: None = Depends(require_admin),
    db: Database = Depends(_require_db),
) -> Dict[str, Any]:
    """Return key metadata and this month's aggregated usage figures.

    The response contains two top-level keys:

    - ``key``: the :class:`~models.APIKeyResponse` metadata.
    - ``usage``: the :class:`~models.UsageSummary` for the current calendar
      month.

    Raises **404** if the key does not exist.
    """
    records = await db.list_api_keys()
    record = next((r for r in records if r["id"] == key_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail=f"API key not found: {key_id!r}")

    summary_raw = await db.get_usage_summary(key_id)
    now = datetime.now(timezone.utc)
    period = f"{now.year}-{now.month:02d}"

    usage = UsageSummary(
        api_key_id=key_id,
        period=period,
        stt_seconds=summary_raw.get("stt_seconds", 0.0),
        tts_seconds=summary_raw.get("tts_seconds", 0.0),
        llm_tokens=summary_raw.get("llm_tokens", 0),
        session_count=summary_raw.get("session_starts", 0),
        monthly_minutes_used=round(
            (summary_raw.get("stt_seconds", 0.0) + summary_raw.get("tts_seconds", 0.0)) / 60.0,
            3,
        ),
        monthly_minutes_limit=record.get("monthly_minutes", 60),
    )

    return {
        "key": _to_api_key_response(record).model_dump(),
        "usage": usage.model_dump(),
    }


@admin_router.delete(
    "/keys/{key_id}",
    status_code=204,
    summary="Deactivate an API key",
)
async def deactivate_api_key(
    key_id: str,
    _: None = Depends(require_admin),
    db: Database = Depends(_require_db),
) -> None:
    """Mark an API key as inactive.

    Deactivated keys are rejected at auth time but **not deleted** from the
    database, so their usage history is preserved.

    Raises **404** if the key does not exist.
    """
    records = await db.list_api_keys()
    record = next((r for r in records if r["id"] == key_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail=f"API key not found: {key_id!r}")

    await db.deactivate_api_key(key_id)
    logger.info(f"Admin: deactivated API key {key_id!r}")


# ---------------------------------------------------------------------------
# Session endpoints
# ---------------------------------------------------------------------------

@admin_router.get(
    "/sessions",
    response_model=List[SessionInfo],
    summary="List active sessions",
)
async def list_sessions(
    _: None = Depends(require_admin),
    db: Database = Depends(_require_db),
) -> List[SessionInfo]:
    """Return all sessions currently marked as connected in the database.

    Results are ordered by most-recent activity first.
    """
    records = await db.list_active_sessions()
    return [_to_session_info(r) for r in records]


@admin_router.get(
    "/sessions/{session_id}",
    response_model=SessionInfo,
    summary="Get session details",
)
async def get_session(
    session_id: str,
    _: None = Depends(require_admin),
    db: Database = Depends(_require_db),
) -> SessionInfo:
    """Fetch metadata for a single session.

    Conversation history is not included in the response; use
    ``history_length`` to gauge conversation depth.

    Raises **404** if the session does not exist.
    """
    record = await db.get_session(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id!r}")
    return _to_session_info(record)


@admin_router.delete(
    "/sessions/{session_id}",
    status_code=204,
    summary="Force-close a session",
)
async def force_close_session(
    session_id: str,
    _: None = Depends(require_admin),
    db: Database = Depends(_require_db),
) -> None:
    """Forcibly disconnect and mark a session as closed.

    If the session has a live WebSocket connection (tracked in the in-memory
    ``_sessions`` dict injected via :func:`init_admin`), the WebSocket is
    closed gracefully before the DB record is updated.

    Raises **404** if no session with *session_id* exists.
    """
    record = await db.get_session(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id!r}")

    # Attempt to close the live WebSocket if the session is in memory.
    if _sessions and session_id in _sessions:
        live_session = _sessions[session_id]
        ws = getattr(live_session, "websocket", None)
        if ws is not None:
            try:
                await ws.close(code=4000, reason="Closed by admin")
            except Exception as exc:
                logger.warning(
                    f"Admin: could not gracefully close WebSocket for {session_id}: {exc}"
                )
        live_session.connected = False
        live_session.websocket = None
        logger.info(f"Admin: force-closed live session {session_id!r}")

    # Mark disconnected in the DB.
    await db.update_session(session_id, is_connected=False)
    logger.info(f"Admin: session {session_id!r} marked disconnected in DB")


# ---------------------------------------------------------------------------
# Usage endpoints
# ---------------------------------------------------------------------------

@admin_router.get(
    "/usage/{key_id}",
    response_model=UsageSummary,
    summary="Usage summary for a key",
)
async def get_usage_summary(
    key_id: str,
    period: Optional[str] = Query(
        None,
        description="Billing period in 'YYYY-MM' format. Defaults to the current month.",
        example="2026-05",
    ),
    _: None = Depends(require_admin),
    db: Database = Depends(_require_db),
) -> UsageSummary:
    """Return aggregated usage statistics for an API key.

    Pass an optional ``?period=YYYY-MM`` query parameter to query a specific
    calendar month.  The default is the current calendar month (UTC).

    Raises **404** if the key does not exist.
    """
    # Verify the key exists
    all_keys = await db.list_api_keys()
    record = next((r for r in all_keys if r["id"] == key_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail=f"API key not found: {key_id!r}")

    # Build period_start from the period string
    period_start: Optional[str] = None
    now = datetime.now(timezone.utc)
    display_period: str

    if period:
        try:
            year, month = period.split("-")
            period_start = datetime(int(year), int(month), 1, tzinfo=timezone.utc).isoformat()
            display_period = period
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=422,
                detail="period must be in 'YYYY-MM' format (e.g. '2026-05')",
            )
    else:
        period_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc).isoformat()
        display_period = f"{now.year}-{now.month:02d}"

    summary_raw = await db.get_usage_summary(key_id, period_start=period_start)

    return UsageSummary(
        api_key_id=key_id,
        period=display_period,
        stt_seconds=summary_raw.get("stt_seconds", 0.0),
        tts_seconds=summary_raw.get("tts_seconds", 0.0),
        llm_tokens=summary_raw.get("llm_tokens", 0),
        session_count=summary_raw.get("session_starts", 0),
        monthly_minutes_used=round(
            (summary_raw.get("stt_seconds", 0.0) + summary_raw.get("tts_seconds", 0.0)) / 60.0,
            3,
        ),
        monthly_minutes_limit=record.get("monthly_minutes", 60),
    )


@admin_router.get(
    "/usage/{key_id}/events",
    response_model=UsageEventsPage,
    summary="Raw usage events for a key (paginated)",
)
async def get_usage_events(
    key_id: str,
    offset: int = Query(0, ge=0, description="Number of events to skip."),
    limit: int = Query(100, ge=1, le=1000, description="Maximum events to return."),
    event_type: Optional[str] = Query(
        None,
        description=(
            "Filter by event type: stt_seconds, tts_seconds, "
            "llm_tokens, session_start, session_end."
        ),
    ),
    _: None = Depends(require_admin),
    db: Database = Depends(_require_db),
) -> UsageEventsPage:
    """Return a paginated list of raw usage events for an API key.

    Events are ordered newest-first.  Use ``offset`` and ``limit`` for
    pagination.

    Raises **404** if the key does not exist.
    """
    all_keys = await db.list_api_keys()
    record = next((r for r in all_keys if r["id"] == key_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail=f"API key not found: {key_id!r}")

    # Fetch events directly from the DB connection.
    conn = db._require_conn()

    # Build query — optionally filter by event_type.
    params: list = [key_id]
    type_clause = ""
    if event_type:
        type_clause = "AND event_type = ?"
        params.append(event_type)

    # Count total rows first.
    count_cursor = await conn.execute(
        f"SELECT COUNT(*) FROM usage_events WHERE api_key_id = ? {type_clause}",
        params,
    )
    count_row = await count_cursor.fetchone()
    total: int = count_row[0] if count_row else 0

    # Fetch paginated rows.
    page_params = params + [limit, offset]
    rows_cursor = await conn.execute(
        f"""
        SELECT id, api_key_id, session_id, event_type, value, created_at
        FROM usage_events
        WHERE api_key_id = ? {type_clause}
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
        """,
        page_params,
    )
    rows = await rows_cursor.fetchall()

    events: List[UsageEvent] = []
    for row in rows:
        r = dict(row)
        dt = _parse_dt(r.get("created_at")) or datetime.now(timezone.utc)
        events.append(
            UsageEvent(
                id=r["id"],
                api_key_id=r["api_key_id"],
                session_id=r.get("session_id"),
                event_type=r["event_type"],
                value=r["value"],
                created_at=dt,
            )
        )

    return UsageEventsPage(
        key_id=key_id,
        total=total,
        offset=offset,
        limit=limit,
        events=events,
    )


# ---------------------------------------------------------------------------
# System stats endpoint
# ---------------------------------------------------------------------------

@admin_router.get(
    "/stats",
    response_model=SystemStats,
    summary="System-wide operational stats",
)
async def get_system_stats(
    _: None = Depends(require_admin),
    db: Database = Depends(_require_db),
) -> SystemStats:
    """Return a snapshot of server-wide metrics.

    Includes session counts, key counts, agent counts, process uptime, and
    the identifiers of the active STT / TTS / VAD backends.

    The STT, TTS, and VAD backend identifiers are read from environment
    variables set by the main application at startup:
    - ``_ADMIN_STT_BACKEND``
    - ``_ADMIN_TTS_BACKEND``
    - ``_ADMIN_VAD_BACKEND``

    These are set automatically when ``init_admin`` is wired up in main.py;
    they fall back to ``"unknown"`` if not configured.
    """
    conn = db._require_conn()

    # Active sessions (DB-level)
    active_cursor = await conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE is_connected = TRUE"
    )
    active_row = await active_cursor.fetchone()
    active_sessions: int = active_row[0] if active_row else 0

    # Total sessions
    total_cursor = await conn.execute("SELECT COUNT(*) FROM sessions")
    total_row = await total_cursor.fetchone()
    total_sessions: int = total_row[0] if total_row else 0

    # API keys (all, active + inactive)
    keys_cursor = await conn.execute("SELECT COUNT(*) FROM api_keys")
    keys_row = await keys_cursor.fetchone()
    total_api_keys: int = keys_row[0] if keys_row else 0

    # Agents
    agents_cursor = await conn.execute("SELECT COUNT(*) FROM agents")
    agents_row = await agents_cursor.fetchone()
    total_agents: int = agents_row[0] if agents_row else 0

    uptime = time.time() - _server_start if _server_start else 0.0

    # Backend identifiers: main.py sets these env vars on startup so that
    # the admin API can report which models are active.
    stt_backend = os.environ.get("_ADMIN_STT_BACKEND", "unknown")
    tts_backend = os.environ.get("_ADMIN_TTS_BACKEND", "unknown")
    vad_backend = os.environ.get("_ADMIN_VAD_BACKEND", "unknown")

    # If the in-memory sessions dict is available, prefer its count as it
    # reflects live WebSocket connections which may differ from the DB flag.
    if _sessions is not None:
        active_sessions = sum(
            1 for s in _sessions.values() if getattr(s, "connected", False)
        )

    return SystemStats(
        active_sessions=active_sessions,
        total_sessions=total_sessions,
        total_api_keys=total_api_keys,
        total_agents=total_agents,
        uptime=round(uptime, 2),
        stt_backend=stt_backend,
        tts_backend=tts_backend,
        vad_backend=vad_backend,
    )
