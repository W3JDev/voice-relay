"""
Tests for the Telnyx adapter.

These tests are deliberately self-contained: they do NOT spin up the full
voice-relay FastAPI app, they only mount ``telnyx_router`` onto a fresh
FastAPI app so each test is isolated.

The ed25519 verification path is tested both happy and unhappy.  Network
calls to Telnyx are intercepted via ``httpx.MockTransport`` -- no real HTTP
traffic is generated.

Run with:

    pytest tests/test_telnyx_adapter.py -v
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any, Dict

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nacl.signing import SigningKey

from src.server import telnyx


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def signing_keypair():
    """Generate a fresh ed25519 keypair for each test."""
    signing_key = SigningKey.generate()
    verify_key = signing_key.verify_key
    public_b64 = base64.b64encode(bytes(verify_key)).decode("ascii")
    return signing_key, public_b64


@pytest.fixture
def configured_adapter(signing_keypair):
    """Configure the adapter with a test key and reset between tests."""
    _, public_b64 = signing_keypair
    telnyx.configure_telnyx(
        api_key="test-api-key",
        public_key=public_b64,
        did="+97140000000",
        voice_relay_ws="wss://voice-relay.example.com/voice/ws",
        hermes_base_url="https://hermes.example.com",
        skip_signature_verification=False,
    )
    telnyx.set_chat_backend(None)
    yield telnyx.get_config()
    # Clean up so other test modules don't see leftover state.
    telnyx.configure_telnyx()


@pytest.fixture
def app_client(configured_adapter):
    """A FastAPI TestClient with just the telnyx_router mounted."""
    app = FastAPI()
    app.include_router(telnyx.telnyx_router)
    return TestClient(app)


def _sign(signing_key: SigningKey, timestamp: str, body: bytes) -> str:
    """Replicate the Telnyx canonical signing string."""
    signed = signing_key.sign(f"{timestamp}|".encode("utf-8") + body)
    return base64.b64encode(signed.signature).decode("ascii")


# ---------------------------------------------------------------------------
# Required tests
# ---------------------------------------------------------------------------


def test_sms_inbound_webhook_valid_signature(
    app_client, signing_keypair, configured_adapter, monkeypatch
):
    """A correctly-signed message.received webhook is accepted, the body is
    parsed, and an outbound SMS is sent via the Telnyx Messages API."""
    signing_key, _ = signing_keypair

    sent_payloads: list[Dict[str, Any]] = []

    async def fake_send_sms(to, from_, text, **kwargs):
        sent_payloads.append({"to": to, "from": from_, "text": text})
        return {"data": {"id": "msg-1", "type": "message"}}

    monkeypatch.setattr(telnyx, "send_sms", fake_send_sms)

    # Stub the chat backend so we never actually call Hermes.
    class _FakeBackend:
        async def chat(self, text, history=None, system_prompt=None):
            return (f"echo: {text}", [])

    telnyx.set_chat_backend(_FakeBackend())

    body_obj = {
        "data": {
            "event_type": "message.received",
            "payload": {
                "id": "abc-123",
                "text": "Hello there",
                "from": {"phone_number": "+15551234567"},
                "to": [{"phone_number": "+97140000000"}],
            },
        }
    }
    body = json.dumps(body_obj).encode("utf-8")
    ts = str(int(time.time()))
    sig = _sign(signing_key, ts, body)

    resp = app_client.post(
        "/webhooks/telnyx",
        content=body,
        headers={
            telnyx.SIG_HEADER: sig,
            telnyx.TS_HEADER: ts,
            "Content-Type": "application/json",
        },
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "received"
    assert resp.json()["reply_status"] == "sent"
    assert sent_payloads == [
        {"to": "+15551234567", "from": "+97140000000", "text": "echo: Hello there"}
    ]


def test_sms_inbound_webhook_invalid_signature_rejected(
    app_client, signing_keypair, configured_adapter
):
    """A webhook signed with the wrong key returns 401."""
    # Sign with a DIFFERENT key, but the adapter only trusts the one we
    # configured in the fixture.
    wrong_key = SigningKey.generate()

    body_obj = {"data": {"event_type": "message.received", "payload": {}}}
    body = json.dumps(body_obj).encode("utf-8")
    ts = str(int(time.time()))
    sig = _sign(wrong_key, ts, body)

    resp = app_client.post(
        "/webhooks/telnyx",
        content=body,
        headers={
            telnyx.SIG_HEADER: sig,
            telnyx.TS_HEADER: ts,
            "Content-Type": "application/json",
        },
    )

    assert resp.status_code == 401
    assert "signature mismatch" in resp.json()["detail"].lower()


def test_sms_inbound_webhook_stale_timestamp_rejected(
    app_client, signing_keypair, configured_adapter
):
    """A webhook with a timestamp outside the freshness window returns 401."""
    signing_key, _ = signing_keypair

    body = b'{"data":{"event_type":"message.received","payload":{}}}'
    # 10 minutes in the past -- past the 5 minute window.
    ts = str(int(time.time()) - 600)
    sig = _sign(signing_key, ts, body)

    resp = app_client.post(
        "/webhooks/telnyx",
        content=body,
        headers={
            telnyx.SIG_HEADER: sig,
            telnyx.TS_HEADER: ts,
            "Content-Type": "application/json",
        },
    )

    assert resp.status_code == 401
    assert "timestamp" in resp.json()["detail"].lower()


def test_sms_inbound_webhook_missing_headers_rejected(app_client, configured_adapter):
    """Missing Telnyx-Signature-Ed25519 / Telnyx-Timestamp returns 401."""
    resp = app_client.post(
        "/webhooks/telnyx",
        content=b'{"data":{}}',
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_outbound_sms_formats_payload_correctly(configured_adapter):
    """send_sms posts the expected JSON body, bearer auth, and URL."""
    captured: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["headers"] = dict(request.headers)
        captured["json"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={"data": {"id": "msg-out-1", "type": "message"}},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        result = await telnyx.send_sms(
            to="+15551234567",
            from_="+97140000000",
            text="Hi from Hermes",
            client=client,
        )

    assert captured["url"] == "https://api.telnyx.com/v2/messages"
    assert captured["method"] == "POST"
    assert captured["headers"]["authorization"] == "Bearer test-api-key"
    assert captured["json"] == {
        "from": "+97140000000",
        "to": "+15551234567",
        "text": "Hi from Hermes",
    }
    assert result["data"]["id"] == "msg-out-1"


def test_call_initiated_answers_call(
    app_client, signing_keypair, configured_adapter, monkeypatch
):
    """A call.initiated Call Control webhook triggers POST .../actions/answer."""
    signing_key, _ = signing_keypair

    issued_commands: list[Dict[str, Any]] = []

    async def fake_call_command(call_control_id, action, *, body=None, **_):
        issued_commands.append(
            {"call_control_id": call_control_id, "action": action, "body": body}
        )
        return {"data": {"result": "ok"}}

    monkeypatch.setattr(telnyx, "_call_command", fake_call_command)

    body_obj = {
        "data": {
            "event_type": "call.initiated",
            "payload": {
                "call_control_id": "ccid-xyz-789",
                "from": "+15551234567",
                "to": "+97140000000",
            },
        }
    }
    body = json.dumps(body_obj).encode("utf-8")
    ts = str(int(time.time()))
    sig = _sign(signing_key, ts, body)

    resp = app_client.post(
        "/webhooks/telnyx/voice",
        content=body,
        headers={
            telnyx.SIG_HEADER: sig,
            telnyx.TS_HEADER: ts,
            "Content-Type": "application/json",
        },
    )

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"status": "answered"}
    assert issued_commands == [
        {"call_control_id": "ccid-xyz-789", "action": "answer", "body": None}
    ]


def test_call_answered_starts_media_stream(
    app_client, signing_keypair, configured_adapter, monkeypatch
):
    """call.answered triggers streaming_start with the configured WS URL."""
    signing_key, _ = signing_keypair

    issued_commands: list[Dict[str, Any]] = []

    async def fake_call_command(call_control_id, action, *, body=None, **_):
        issued_commands.append(
            {"call_control_id": call_control_id, "action": action, "body": body}
        )
        return {"data": {"result": "ok"}}

    monkeypatch.setattr(telnyx, "_call_command", fake_call_command)

    body_obj = {
        "data": {
            "event_type": "call.answered",
            "payload": {"call_control_id": "ccid-xyz-789"},
        }
    }
    body = json.dumps(body_obj).encode("utf-8")
    ts = str(int(time.time()))
    sig = _sign(signing_key, ts, body)

    resp = app_client.post(
        "/webhooks/telnyx/voice",
        content=body,
        headers={
            telnyx.SIG_HEADER: sig,
            telnyx.TS_HEADER: ts,
            "Content-Type": "application/json",
        },
    )

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"status": "streaming"}
    assert len(issued_commands) == 1
    cmd = issued_commands[0]
    assert cmd["action"] == "streaming_start"
    assert cmd["body"]["stream_url"] == "wss://voice-relay.example.com/voice/ws"
    assert cmd["body"]["stream_bidirectional_codec"] == "L16"


def test_verify_ed25519_signature_unit(signing_keypair):
    """Direct unit test of verify_ed25519_signature -- happy + unhappy paths."""
    signing_key, public_b64 = signing_keypair

    body = b'{"hello":"world"}'
    ts = str(int(time.time()))
    sig = _sign(signing_key, ts, body)

    assert telnyx.verify_ed25519_signature(public_b64, ts, body, sig) is True

    # Tampered body fails.
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        telnyx.verify_ed25519_signature(public_b64, ts, b'{"hello":"evil"}', sig)
    assert excinfo.value.status_code == 401
