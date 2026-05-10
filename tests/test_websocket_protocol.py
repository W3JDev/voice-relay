"""
Voice Relay — WebSocket Protocol Test Suite
Phase 1 QA: validates message types, session isolation, barge-in, and streaming STT.

Usage:
    pytest tests/test_websocket_protocol.py -v
    pytest tests/test_websocket_protocol.py -v -k test_session  # just session tests

Requires:
    VOICE_RELAY_URL env var (e.g. ws://localhost:8765/ws)
"""

import asyncio
import base64
import json
import os
import struct
import uuid

import numpy as np
import pytest
import websockets


RELAY_URL = os.environ.get("VOICE_RELAY_URL", "ws://localhost:8765/ws")
RELAY_HTTP = RELAY_URL.replace("ws://", "http://").replace("/ws", "")


# ─── Helpers ───

def generate_silence(duration_s: float = 0.5, sample_rate: int = 16000) -> bytes:
    """Generate silent audio as base64-encoded float32 PCM."""
    samples = int(duration_s * sample_rate)
    audio = np.zeros(samples, dtype=np.float32)
    return base64.b64encode(audio.tobytes()).decode()


def generate_tone(freq: float = 440, duration_s: float = 0.5,
                  sample_rate: int = 16000, amplitude: float = 0.5) -> bytes:
    """Generate a sine wave tone as base64-encoded float32 PCM."""
    t = np.linspace(0, duration_s, int(duration_s * sample_rate), endpoint=False)
    audio = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return base64.b64encode(audio.tobytes()).decode()


async def connect(url: str = RELAY_URL, timeout: float = 5.0):
    """Connect to the voice relay and wait for session_start."""
    ws = await asyncio.wait_for(
        websockets.connect(url, ping_interval=None),
        timeout=timeout,
    )
    # Wait for session_start message
    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
    msg = json.loads(raw)
    assert msg["type"] == "session_start", f"Expected session_start, got {msg['type']}"
    assert "session_id" in msg, "session_start missing session_id"
    return ws, msg["session_id"]


async def recv_until(ws, msg_type: str, timeout: float = 10.0):
    """Receive messages until we get one of the specified type."""
    deadline = asyncio.get_event_loop().time() + timeout
    messages = []
    while asyncio.get_event_loop().time() < deadline:
        remaining = deadline - asyncio.get_event_loop().time()
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            msg = json.loads(raw)
            messages.append(msg)
            if msg["type"] == msg_type:
                return msg, messages
        except asyncio.TimeoutError:
            break
    pytest.fail(f"Never received {msg_type}. Got: {[m['type'] for m in messages]}")


# ─── Test: Health Endpoint ───

@pytest.mark.asyncio
async def test_health_endpoint():
    """GET /health returns status ok."""
    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{RELAY_HTTP}/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "active_sessions" in data
        assert "uptime" in data


# ─── Test: Session Lifecycle ───

@pytest.mark.asyncio
async def test_session_start():
    """Connecting should return a session_start message with a UUID session_id."""
    ws, session_id = await connect()
    # session_id should be a valid UUID
    uuid.UUID(session_id)  # raises ValueError if invalid
    await ws.close()


@pytest.mark.asyncio
async def test_session_isolation():
    """Two concurrent connections should get different session_ids."""
    ws1, sid1 = await connect()
    ws2, sid2 = await connect()
    assert sid1 != sid2, "Sessions should be unique"
    await ws1.close()
    await ws2.close()


@pytest.mark.asyncio
async def test_session_reconnect():
    """Reconnecting with a session_id should restore the session."""
    ws1, sid = await connect()
    await ws1.close()

    # Small delay to simulate network blip
    await asyncio.sleep(0.2)

    # Reconnect with session_id
    ws2 = await asyncio.wait_for(
        websockets.connect(RELAY_URL, ping_interval=None),
        timeout=5.0,
    )
    # Consume the automatic session_start that fires on every connection
    init_raw = await asyncio.wait_for(ws2.recv(), timeout=5.0)
    init_msg = json.loads(init_raw)
    assert init_msg["type"] == "session_start", "Should get initial session_start"

    # Now request reconnection with the old session id
    await ws2.send(json.dumps({"type": "reconnect", "session_id": sid}))
    raw = await asyncio.wait_for(ws2.recv(), timeout=5.0)
    msg = json.loads(raw)
    assert msg["type"] == "session_start"
    assert msg.get("resumed") is True, "Should indicate resumed session"
    assert msg["session_id"] == sid, "Should restore same session"
    await ws2.close()


# ─── Test: Audio Pipeline ───

@pytest.mark.asyncio
async def test_audio_send_receive_vad():
    """Sending audio should trigger VAD status responses."""
    ws, sid = await connect()

    # Send a tone chunk (simulates speech)
    tone = generate_tone(freq=440, duration_s=0.3)
    await ws.send(json.dumps({"type": "audio", "data": tone}))

    # Should get a vad_status back
    msg, _ = await recv_until(ws, "vad_status", timeout=5.0)
    assert "speech_detected" in msg

    await ws.close()


@pytest.mark.asyncio
async def test_silence_no_transcript():
    """Sending only silence should not produce a transcript."""
    ws, sid = await connect()

    # Send silence
    silence = generate_silence(duration_s=2.0)
    await ws.send(json.dumps({"type": "audio", "data": silence}))

    # Wait briefly — should not get a transcript
    try:
        raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
        msg = json.loads(raw)
        # We might get vad_status but should NOT get a transcript
        if msg["type"] == "transcript":
            pytest.fail("Got transcript from pure silence")
    except asyncio.TimeoutError:
        pass  # Expected — no transcript from silence

    await ws.close()


# ─── Test: Ping/Pong ───

@pytest.mark.asyncio
async def test_ping_pong():
    """Server should respond to ping with pong."""
    ws, sid = await connect()
    await ws.send(json.dumps({"type": "ping"}))

    msg, _ = await recv_until(ws, "pong", timeout=5.0)
    assert msg["type"] == "pong"

    await ws.close()


# ─── Test: Message Format Validation ───

@pytest.mark.asyncio
async def test_invalid_json():
    """Sending invalid JSON should not crash the server."""
    ws, sid = await connect()
    await ws.send("this is not json{{{")

    # Server should either send an error or just ignore
    # The connection should stay alive
    await ws.send(json.dumps({"type": "ping"}))
    msg, _ = await recv_until(ws, "pong", timeout=5.0)
    assert msg["type"] == "pong"

    await ws.close()


@pytest.mark.asyncio
async def test_unknown_message_type():
    """Sending an unknown message type should not crash."""
    ws, sid = await connect()
    await ws.send(json.dumps({"type": "nonexistent_type", "data": "test"}))

    # Connection should survive
    await ws.send(json.dumps({"type": "ping"}))
    msg, _ = await recv_until(ws, "pong", timeout=5.0)
    assert msg["type"] == "pong"

    await ws.close()


# ─── Test: Concurrent Sessions Under Load ───

@pytest.mark.asyncio
async def test_concurrent_sessions():
    """Multiple concurrent connections should all work independently."""
    sessions = []
    for _ in range(5):
        ws, sid = await connect()
        sessions.append((ws, sid))

    # All should have unique IDs
    sids = [s[1] for s in sessions]
    assert len(set(sids)) == 5, "All session IDs should be unique"

    # All should respond to ping
    for ws, sid in sessions:
        await ws.send(json.dumps({"type": "ping"}))

    for ws, sid in sessions:
        msg, _ = await recv_until(ws, "pong", timeout=5.0)
        assert msg["type"] == "pong"

    for ws, _ in sessions:
        await ws.close()
