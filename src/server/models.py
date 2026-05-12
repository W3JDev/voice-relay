"""
Pydantic models for the Voice Relay server -- Phase 2.

Covers:
  - Agent backend configuration
  - API key management (create / response)
  - Usage summaries
  - Session info (admin API)
  - System-wide stats

All models use Pydantic v2 conventions (``model_config``, ``model_validate``,
etc.).  Datetime fields are always timezone-aware UTC.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Agent configuration
# ---------------------------------------------------------------------------

class AgentConfig(BaseModel):
    """
    Configuration for an agent backend that sessions can route to.

    Agents wrap an OpenAI-compatible (or OpenClaw) LLM endpoint and expose a
    stable ``id`` slug that API keys and sessions reference.
    """

    id: str = Field(
        ...,
        description="Unique slug for the agent, e.g. 'hermes' or 'custom-1'.",
        examples=["hermes", "openclaw", "custom-1"],
    )
    name: str = Field(
        ...,
        description="Human-readable display name.",
        examples=["Hermes Agent", "OpenClaw Voice"],
    )
    backend_type: str = Field(
        default="openai",
        description="Backend protocol to use: 'openai' or 'openclaw'.",
        examples=["openai", "openclaw"],
    )
    url: str = Field(
        ...,
        description="Base URL of the OpenAI-compatible endpoint (without /chat/completions).",
        examples=["https://hermes-agent.railway.internal/v1"],
    )
    model: str = Field(
        ...,
        description="Model identifier passed to the backend.",
        examples=["hermes-agent", "gpt-4o-mini"],
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for the backend endpoint.  May be encrypted at rest.",
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system prompt override for this agent.",
    )
    voice: Optional[str] = Field(
        default=None,
        description="Kokoro TTS voice ID for this agent (e.g. 'am_adam', 'af_bella'). NULL uses server default.",
    )
    is_default: bool = Field(
        default=False,
        description="Whether this agent is the system-wide default for new sessions.",
    )

    @field_validator("backend_type")
    @classmethod
    def validate_backend_type(cls, v: str) -> str:
        """Ensure backend_type is one of the supported values."""
        allowed = {"openai", "openclaw"}
        if v not in allowed:
            raise ValueError(f"backend_type must be one of {sorted(allowed)}, got {v!r}")
        return v


# ---------------------------------------------------------------------------
# API key management
# ---------------------------------------------------------------------------

class APIKeyCreate(BaseModel):
    """
    Request body for creating a new API key.

    Submitted to the key-management endpoint by an administrator.
    """

    name: str = Field(
        ...,
        description="Human-readable label for this API key.",
        examples=["My App Production Key"],
    )
    tier: str = Field(
        default="free",
        description="Pricing tier: 'free', 'pro', or 'enterprise'.",
        examples=["free", "pro", "enterprise"],
    )
    agent_id: Optional[str] = Field(
        default=None,
        description=(
            "Default agent slug for sessions created with this key.  "
            "NULL means the system default agent is used."
        ),
        examples=["hermes", None],
    )
    rate_limit_per_minute: int = Field(
        default=10,
        ge=1,
        description="Maximum number of authenticated requests allowed per minute.",
        examples=[10, 60, 120],
    )
    monthly_minutes: int = Field(
        default=60,
        ge=0,
        description="Monthly audio-minute quota (STT + TTS combined).  0 = unlimited.",
        examples=[60, 500],
    )
    expires_at: Optional[datetime] = Field(
        default=None,
        description="Optional UTC datetime after which this key is no longer valid.",
    )

    @field_validator("tier")
    @classmethod
    def validate_tier(cls, v: str) -> str:
        """Ensure tier is one of the accepted values."""
        allowed = {"free", "pro", "enterprise"}
        if v not in allowed:
            raise ValueError(f"tier must be one of {sorted(allowed)}, got {v!r}")
        return v


class APIKeyResponse(BaseModel):
    """
    Public representation of an API key record.

    The plaintext key is **never** included in this model.  Use ``key_prefix``
    for identification in logs and admin UIs.
    """

    id: str = Field(
        ...,
        description="UUID primary key of this record.",
    )
    key_prefix: str = Field(
        ...,
        description="First characters of the key for display (e.g. 'vr_live_a3').",
        examples=["vr_live_a3"],
    )
    name: str = Field(
        ...,
        description="Human-readable label.",
    )
    tier: str = Field(
        ...,
        description="Pricing tier of this key.",
    )
    rate_limit_per_minute: int = Field(
        ...,
        description="Requests-per-minute cap.",
    )
    monthly_minutes: int = Field(
        ...,
        description="Monthly audio-minute quota.",
    )
    is_active: bool = Field(
        ...,
        description="False if the key has been deactivated.",
    )
    agent_id: Optional[str] = Field(
        ...,
        description="Default agent slug for this key, or None for the system default.",
    )
    created_at: datetime = Field(
        ...,
        description="UTC datetime when this key was created.",
    )
    expires_at: Optional[datetime] = Field(
        ...,
        description="UTC datetime after which this key expires, or None if it never expires.",
    )


# ---------------------------------------------------------------------------
# Usage tracking
# ---------------------------------------------------------------------------

class UsageSummary(BaseModel):
    """
    Aggregated usage statistics for an API key over a billing period.

    Returned by the usage-summary endpoint.  All second/minute values are
    accumulated from raw ``usage_events`` rows in the database.
    """

    api_key_id: str = Field(
        ...,
        description="UUID of the API key these stats belong to.",
    )
    period: str = Field(
        ...,
        description="Billing period in 'YYYY-MM' format.",
        examples=["2026-05"],
    )
    stt_seconds: float = Field(
        default=0.0,
        ge=0,
        description="Total seconds of audio transcribed by STT this period.",
    )
    tts_seconds: float = Field(
        default=0.0,
        ge=0,
        description="Total seconds of audio synthesised by TTS this period.",
    )
    llm_tokens: int = Field(
        default=0,
        ge=0,
        description="Total LLM tokens charged this period.",
    )
    session_count: int = Field(
        default=0,
        ge=0,
        description="Number of sessions started this period.",
    )
    monthly_minutes_used: float = Field(
        default=0.0,
        ge=0,
        description="(stt_seconds + tts_seconds) / 60 — used against the monthly quota.",
    )
    monthly_minutes_limit: int = Field(
        default=60,
        ge=0,
        description="Monthly minute quota for this API key.  0 means unlimited.",
    )


# ---------------------------------------------------------------------------
# Session info
# ---------------------------------------------------------------------------

class SessionInfo(BaseModel):
    """
    Lightweight session summary for the admin API.

    Full conversation history is excluded for bandwidth reasons.  Use
    ``history_length`` to see how many turns exist without transferring them.
    """

    id: str = Field(
        ...,
        description="UUID of the session.",
    )
    api_key_id: Optional[str] = Field(
        ...,
        description="UUID of the API key that created this session, or None.",
    )
    agent_id: Optional[str] = Field(
        ...,
        description="Slug of the agent backend assigned to this session, or None.",
    )
    is_connected: bool = Field(
        ...,
        description="Whether the client WebSocket is currently connected.",
    )
    history_length: int = Field(
        ...,
        ge=0,
        description="Number of messages (user + assistant turns) in the conversation history.",
    )
    created_at: datetime = Field(
        ...,
        description="UTC datetime when this session was first created.",
    )
    last_activity: datetime = Field(
        ...,
        description="UTC datetime of the last message or heartbeat.",
    )


# ---------------------------------------------------------------------------
# System-wide admin stats
# ---------------------------------------------------------------------------

class SystemStats(BaseModel):
    """
    Snapshot of server-wide operational metrics.

    Returned by the ``GET /admin/stats`` endpoint.  Intended for dashboards
    and monitoring integrations.
    """

    active_sessions: int = Field(
        ...,
        ge=0,
        description="Number of sessions with a live WebSocket connection right now.",
    )
    total_sessions: int = Field(
        ...,
        ge=0,
        description="Total sessions in the database (connected + disconnected).",
    )
    total_api_keys: int = Field(
        ...,
        ge=0,
        description="Total API key records in the database (active + inactive).",
    )
    total_agents: int = Field(
        ...,
        ge=0,
        description="Number of registered agent backend configurations.",
    )
    uptime: float = Field(
        ...,
        ge=0,
        description="Server process uptime in seconds.",
    )
    stt_backend: str = Field(
        ...,
        description="Identifier of the active STT backend (e.g. 'whisper-base', 'vsaas').",
        examples=["whisper-base", "vsaas"],
    )
    tts_backend: str = Field(
        ...,
        description="Identifier of the active TTS backend (e.g. 'chatterbox', 'edge-tts').",
        examples=["chatterbox", "edge-tts"],
    )
    vad_backend: str = Field(
        ...,
        description="Identifier of the active VAD backend (e.g. 'silero', 'webrtc').",
        examples=["silero", "webrtc"],
    )
