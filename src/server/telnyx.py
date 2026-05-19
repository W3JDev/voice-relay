"""
Telnyx adapter for the voice-relay gateway.

This module provides the carrier integration glue for Telnyx:

  * Inbound SMS webhook  -> route into the conversation engine (AIBackend /
    OpenCLAW Gateway / configured OpenAI-compatible upstream) and reply with
    an outbound SMS via the Telnyx Messages API.
  * Outbound SMS helper  -> send a one-off SMS programmatically.
  * Telnyx Call Control webhook -> answer inbound calls, bridge audio to the
    voice-relay WebSocket via the Telnyx Media Streaming command, and clean
    up on hangup.  Audio itself is handled by voice-relay's existing
    /voice/ws WebSocket -- this adapter only emits Call Control commands.

Architecture
------------

    Caller --PSTN--> Telnyx --webhook--> /webhooks/telnyx/voice
                                           |
                                           +-- streaming_start --> wss://<voice-relay>/voice/ws
                                                                       (existing handler does STT, TTS, VAD, barge-in)
                                                                       and calls Hermes /v1/chat/completions

    SMSer  --SMS-->  Telnyx --webhook--> /webhooks/telnyx  -> backend.chat() -> Telnyx Messages API

Security
--------

Telnyx signs every webhook with ed25519 (NOT HMAC-SHA256).  Headers:

    Telnyx-Signature-Ed25519: <base64 signature>
    Telnyx-Timestamp:         <unix timestamp seconds>

Verification: ed25519.verify(public_key, f"{timestamp}|{raw_body}", signature)

The public key is the one Telnyx exposes in the portal under Mission Control
Portal -> Account -> Public Key.  Configure it via the TELNYX_PUBLIC_KEY env
var as the raw base64 string Telnyx provides.

Environment variables
---------------------

    TELNYX_API_KEY        Bearer token for the Telnyx REST API.
    TELNYX_PUBLIC_KEY     Base64-encoded ed25519 public key (32 bytes raw).
    TELNYX_DID            E.164 outbound number (e.g. the +971 4 Dubai DID).
    TELNYX_VOICE_RELAY_WS WebSocket URL voice-relay exposes for inbound media
                          streaming (e.g. wss://voice-relay.example.com/voice/ws).
    HERMES_BASE_URL       (Optional) Override the upstream chat endpoint when
                          the adapter is wired up directly without the
                          AgentRouter.  Defaults to the env-configured backend.
"""

from __future__ import annotations

import base64
import os
import time
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, Header, HTTPException, Request, status
from loguru import logger
from pydantic import BaseModel

try:
    # PyNaCl is the canonical ed25519 implementation in the Python ecosystem
    # and is small / wheel-distributed for every supported platform.
    from nacl.exceptions import BadSignatureError
    from nacl.signing import VerifyKey
except ImportError:  # pragma: no cover - import guard
    VerifyKey = None  # type: ignore[assignment]
    BadSignatureError = Exception  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TELNYX_API_BASE = "https://api.telnyx.com/v2"
TELNYX_MESSAGES_ENDPOINT = f"{TELNYX_API_BASE}/messages"
TELNYX_CALLS_ENDPOINT_TEMPLATE = f"{TELNYX_API_BASE}/calls/{{call_control_id}}/actions/{{action}}"

# Webhook timestamps older than this many seconds are rejected to prevent
# replay.  Telnyx recommends 300s; we mirror that.
WEBHOOK_MAX_AGE_SECONDS = 300

SIG_HEADER = "Telnyx-Signature-Ed25519"
TS_HEADER = "Telnyx-Timestamp"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TelnyxConfig(BaseModel):
    """Configuration for the Telnyx adapter.

    Constructed once at startup from environment variables.  Held by the
    module-level singleton ``_config`` and re-readable in tests via
    ``configure_telnyx()``.
    """

    api_key: Optional[str] = None
    public_key: Optional[str] = None
    did: Optional[str] = None
    voice_relay_ws: Optional[str] = None
    hermes_base_url: Optional[str] = None
    # Skip ed25519 verification.  Intended only for local development and
    # tests; production deployments must keep this False.
    skip_signature_verification: bool = False


_config: TelnyxConfig = TelnyxConfig()


def configure_telnyx(
    *,
    api_key: Optional[str] = None,
    public_key: Optional[str] = None,
    did: Optional[str] = None,
    voice_relay_ws: Optional[str] = None,
    hermes_base_url: Optional[str] = None,
    skip_signature_verification: bool = False,
) -> TelnyxConfig:
    """Replace the module-level config (used at app startup and in tests)."""
    global _config
    _config = TelnyxConfig(
        api_key=api_key,
        public_key=public_key,
        did=did,
        voice_relay_ws=voice_relay_ws,
        hermes_base_url=hermes_base_url,
        skip_signature_verification=skip_signature_verification,
    )
    logger.info(
        "Telnyx adapter configured  (did={did}, voice_relay_ws={ws}, "
        "skip_sig_verify={skip})",
        did=_config.did,
        ws=_config.voice_relay_ws,
        skip=_config.skip_signature_verification,
    )
    return _config


def configure_from_env() -> TelnyxConfig:
    """Load Telnyx configuration from environment variables."""
    return configure_telnyx(
        api_key=os.getenv("TELNYX_API_KEY"),
        public_key=os.getenv("TELNYX_PUBLIC_KEY"),
        did=os.getenv("TELNYX_DID"),
        voice_relay_ws=os.getenv("TELNYX_VOICE_RELAY_WS"),
        hermes_base_url=os.getenv("HERMES_BASE_URL"),
        skip_signature_verification=os.getenv(
            "TELNYX_SKIP_SIGNATURE_VERIFICATION", "false"
        ).lower()
        in ("1", "true", "yes"),
    )


def get_config() -> TelnyxConfig:
    """Return the current adapter config (test hook)."""
    return _config


# ---------------------------------------------------------------------------
# Signature verification
# ---------------------------------------------------------------------------


def verify_ed25519_signature(
    public_key_b64: str,
    timestamp: str,
    body: bytes,
    signature_b64: str,
    *,
    now: Optional[float] = None,
    max_age_seconds: int = WEBHOOK_MAX_AGE_SECONDS,
) -> bool:
    """Verify a Telnyx webhook signature.

    Telnyx signs the concatenation ``f"{timestamp}|{raw_body}"`` (note the
    pipe separator -- it is NOT a dot like Stripe) with the account's
    ed25519 private key.  We verify with the matching public key.

    Returns True on success; raises HTTPException(401) otherwise so that
    FastAPI handlers can simply call this and propagate the rejection.
    """
    if VerifyKey is None:
        raise RuntimeError(
            "PyNaCl is not installed; add `pynacl>=1.5.0` to requirements.txt"
        )

    # 1. Reject stale timestamps to prevent replay.
    try:
        ts_int = int(timestamp)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Telnyx-Timestamp header",
        )

    current = now if now is not None else time.time()
    if abs(current - ts_int) > max_age_seconds:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Telnyx webhook timestamp outside acceptable window",
        )

    # 2. Decode keys + signature.
    try:
        verify_key = VerifyKey(base64.b64decode(public_key_b64))
        signature = base64.b64decode(signature_b64)
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Malformed Telnyx signature material: {exc}",
        )

    # 3. Verify the canonical signed string.
    signed_payload = f"{timestamp}|".encode("utf-8") + body
    try:
        verify_key.verify(signed_payload, signature)
    except BadSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Telnyx webhook signature mismatch",
        )

    return True


async def _verify_request(request: Request, body: bytes) -> None:
    """Pull headers off a FastAPI request and verify the signature."""
    cfg = get_config()
    if cfg.skip_signature_verification:
        logger.warning(
            "Telnyx signature verification SKIPPED -- do not run this in production"
        )
        return

    if not cfg.public_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="TELNYX_PUBLIC_KEY not configured",
        )

    signature = request.headers.get(SIG_HEADER)
    timestamp = request.headers.get(TS_HEADER)
    if not signature or not timestamp:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing {SIG_HEADER} / {TS_HEADER} headers",
        )

    verify_ed25519_signature(cfg.public_key, timestamp, body, signature)


# ---------------------------------------------------------------------------
# Outbound helpers
# ---------------------------------------------------------------------------


async def send_sms(
    to: str,
    from_: str,
    text: str,
    *,
    api_key: Optional[str] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> Dict[str, Any]:
    """Send an outbound SMS via the Telnyx Messages API.

    Args:
        to:      Destination phone number in E.164 format (e.g. "+14155551234").
        from_:   Sender phone number / messaging profile alphanumeric sender.
        text:    Message body.  Telnyx will segment automatically.
        api_key: Override the configured key (used in tests).
        client:  Inject an httpx.AsyncClient to share connection pooling and
                 to make the function trivially mockable from tests.

    Returns:
        The decoded JSON response from Telnyx (typically the created
        ``message.created`` resource under ``{"data": {...}}``).
    """
    key = api_key or get_config().api_key
    if not key:
        raise RuntimeError("TELNYX_API_KEY not configured")

    payload = {"from": from_, "to": to, "text": text}
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=10.0)
    try:
        response = await client.post(
            TELNYX_MESSAGES_ENDPOINT, json=payload, headers=headers
        )
    finally:
        if owns_client:
            await client.aclose()

    if response.status_code >= 400:
        logger.error(
            "Telnyx send_sms failed  status={status} body={body}",
            status=response.status_code,
            body=response.text,
        )
        response.raise_for_status()

    return response.json()


async def _call_command(
    call_control_id: str,
    action: str,
    *,
    body: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> Dict[str, Any]:
    """POST a Telnyx Call Control action (answer / hangup / streaming_start)."""
    key = api_key or get_config().api_key
    if not key:
        raise RuntimeError("TELNYX_API_KEY not configured")

    url = TELNYX_CALLS_ENDPOINT_TEMPLATE.format(
        call_control_id=call_control_id, action=action
    )
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=10.0)
    try:
        response = await client.post(url, json=body or {}, headers=headers)
    finally:
        if owns_client:
            await client.aclose()

    if response.status_code >= 400:
        logger.error(
            "Telnyx call action {action} failed  status={status} body={body}",
            action=action,
            status=response.status_code,
            body=response.text,
        )
        response.raise_for_status()

    return response.json()


# ---------------------------------------------------------------------------
# Conversation engine bridge
# ---------------------------------------------------------------------------


async def _route_to_chat_engine(
    text: str,
    *,
    from_number: str,
    message_id: str,
    backend: Optional[Any] = None,
) -> str:
    """Route an inbound SMS into the conversation engine and return a reply.

    The caller can pass an already-constructed ``backend`` (e.g. the one
    main.py wires up at startup).  When omitted, the function falls back to
    calling HERMES_BASE_URL directly as an OpenAI-compatible
    /v1/chat/completions endpoint.

    SMS sessions are stateless on the carrier side, so we treat each inbound
    text as a single-turn exchange.  Multi-turn memory belongs to whatever
    component (AgentRouter / Hermes) owns conversation state.
    """
    if backend is not None and hasattr(backend, "chat"):
        try:
            reply, _ = await backend.chat(text)
            return reply
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("backend.chat failed for SMS from %s: %s", from_number, exc)
            return "Sorry, I had trouble processing that. Could you try again?"

    cfg = get_config()
    base = cfg.hermes_base_url or os.getenv("HERMES_BASE_URL")
    if not base:
        # Echo as a graceful degraded mode -- matches AIBackend's behaviour
        # when no API key is configured.
        return f"I heard you say: {text}"

    payload = {
        "model": os.getenv("HERMES_MODEL", "default"),
        "messages": [{"role": "user", "content": text}],
        "max_tokens": 500,
        "temperature": 0.7,
        "metadata": {
            "channel": "sms",
            "provider": "telnyx",
            "from": from_number,
            "message_id": message_id,
        },
    }
    headers = {"Content-Type": "application/json"}
    token = os.getenv("HERMES_API_KEY")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            f"{base.rstrip('/')}/v1/chat/completions", json=payload, headers=headers
        )
        resp.raise_for_status()
        body = resp.json()
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        logger.error("Unexpected Hermes response shape: %r", body)
        return "Sorry, something went wrong."


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------


telnyx_router = APIRouter(tags=["telnyx"])

# A reference to the AIBackend (or whatever the host app wires up) that should
# handle inbound SMS.  Set this at startup with ``set_chat_backend``.
_chat_backend: Optional[Any] = None


def set_chat_backend(backend: Any) -> None:
    """Inject the chat backend that inbound SMS should route into."""
    global _chat_backend
    _chat_backend = backend


@telnyx_router.post("/webhooks/telnyx", status_code=status.HTTP_200_OK)
async def telnyx_sms_webhook(request: Request) -> Dict[str, Any]:
    """Inbound SMS webhook (Telnyx Messaging Profile -> Webhook URL).

    Telnyx will retry non-2xx responses up to 10 times with exponential
    backoff, so this handler returns 200 even when downstream processing
    fails (errors are logged).
    """
    body = await request.body()
    await _verify_request(request, body)

    payload = await request.json()
    event_type = (payload.get("data") or {}).get("event_type")
    record = (payload.get("data") or {}).get("payload") or {}

    if event_type != "message.received":
        logger.debug("Ignoring Telnyx SMS event_type={evt}", evt=event_type)
        return {"status": "ignored", "event_type": event_type}

    text = record.get("text", "") or ""
    message_id = record.get("id", "")
    to_obj = record.get("to") or []
    from_obj = record.get("from") or {}
    to_number = (to_obj[0] if to_obj else {}).get("phone_number") or ""
    from_number = from_obj.get("phone_number", "")

    if not from_number or not to_number:
        logger.warning("Telnyx SMS missing from/to: payload=%r", record)
        return {"status": "ignored", "reason": "missing-numbers"}

    logger.info(
        "Inbound SMS  from={from_} to={to} id={mid} text={t!r}",
        from_=from_number,
        to=to_number,
        mid=message_id,
        t=text[:200],
    )

    reply = await _route_to_chat_engine(
        text,
        from_number=from_number,
        message_id=message_id,
        backend=_chat_backend,
    )

    try:
        await send_sms(to=from_number, from_=to_number, text=reply)
    except Exception as exc:
        logger.exception("Failed to send Telnyx SMS reply: %s", exc)
        return {"status": "received", "reply_status": "failed"}

    return {"status": "received", "reply_status": "sent"}


@telnyx_router.post("/webhooks/telnyx/voice", status_code=status.HTTP_200_OK)
async def telnyx_voice_webhook(request: Request) -> Dict[str, Any]:
    """Call Control webhook.

    Hermes never touches audio; we just orchestrate the call lifecycle and
    hand the media stream off to voice-relay's existing WebSocket.
    """
    body = await request.body()
    await _verify_request(request, body)

    payload = await request.json()
    data = payload.get("data") or {}
    event_type = data.get("event_type", "")
    record = data.get("payload") or {}
    call_control_id = record.get("call_control_id", "")

    logger.info(
        "Call Control event  type={evt} call_control_id={ccid}",
        evt=event_type,
        ccid=call_control_id,
    )

    if not call_control_id and event_type not in (
        "streaming.started",
        "streaming.stopped",
    ):
        logger.warning("Call Control event missing call_control_id: %r", record)
        return {"status": "ignored", "reason": "no-call-control-id"}

    cfg = get_config()

    if event_type == "call.initiated":
        # Answer immediately so we can attach the media stream.
        await _call_command(call_control_id, "answer")
        return {"status": "answered"}

    if event_type == "call.answered":
        # Bridge the call into voice-relay's WebSocket.  The codec must match
        # voice-relay's expectation -- we use L16 PCM 16-bit 8kHz, which is
        # what Telnyx Media Streaming supports bidirectionally.
        ws_url = cfg.voice_relay_ws
        if not ws_url:
            logger.error("TELNYX_VOICE_RELAY_WS not configured; cannot stream")
            return {"status": "answered", "stream_status": "skipped"}
        await _call_command(
            call_control_id,
            "streaming_start",
            body={
                "stream_url": ws_url,
                "stream_track": "both_tracks",
                "stream_bidirectional_mode": "rtp",
                "stream_bidirectional_codec": "L16",
            },
        )
        return {"status": "streaming"}

    if event_type == "call.hangup":
        # Telnyx tears the stream down for us; nothing to do but log.
        return {"status": "hung_up"}

    if event_type in ("streaming.started", "streaming.stopped"):
        return {"status": "ack", "event_type": event_type}

    logger.debug("Unhandled Call Control event type: {evt}", evt=event_type)
    return {"status": "ignored", "event_type": event_type}


# Convenience re-exports for tests / external wiring.
__all__ = [
    "telnyx_router",
    "send_sms",
    "verify_ed25519_signature",
    "configure_telnyx",
    "configure_from_env",
    "get_config",
    "set_chat_backend",
    "TelnyxConfig",
]
