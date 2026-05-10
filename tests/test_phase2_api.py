"""
Phase 2 API tests for the Voice Relay admin endpoints.

Tests the admin REST API (agents, keys, sessions, stats) using
FastAPI's TestClient with an in-memory SQLite database.

Requires:
    pip install pytest pytest-asyncio httpx
"""

import os
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Set up env vars BEFORE importing the application modules, so that the
# admin key guard and database path are configured for testing.
# ---------------------------------------------------------------------------

MASTER_KEY = "test-master-key-for-phase2"
os.environ["OPENCLAW_MASTER_KEY"] = MASTER_KEY
os.environ["OPENCLAW_DB_PATH"] = ":memory:"
os.environ["OPENCLAW_REQUIRE_AUTH"] = "false"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _patched_app():
    """
    Build a TestClient around the real FastAPI app, but stub out the
    heavy ML models (STT, TTS, VAD) so the test suite stays fast and
    does not require GPU / large model downloads.
    """
    # Patch the heavyweight imports before main.py tries to load them.
    mock_stt = MagicMock()
    mock_stt.backend = "mock"
    mock_stt.transcribe = AsyncMock(return_value="hello")

    mock_tts = MagicMock()
    mock_tts.backend = "mock"

    async def _tts_stream(text):
        yield b"\x00" * 480
    mock_tts.synthesize_stream = _tts_stream

    mock_vad = MagicMock()
    mock_vad.backend = "mock"
    mock_vad.is_speech = MagicMock(return_value=False)

    with patch("src.server.main.WhisperSTT", return_value=mock_stt), \
         patch("src.server.main.RelayTTS", return_value=mock_tts), \
         patch("src.server.main.VoiceActivityDetector", return_value=mock_vad):

        from src.server.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            yield client


@pytest.fixture(scope="module")
def client(_patched_app):
    return _patched_app


@pytest.fixture(scope="module")
def admin_headers():
    """Standard headers for admin-authenticated requests."""
    return {"Authorization": f"Bearer {MASTER_KEY}"}


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "db_connected" in body
        assert body["db_connected"] is True

    def test_health_contains_uptime(self, client):
        resp = client.get("/health")
        body = resp.json()
        assert "uptime" in body
        assert isinstance(body["uptime"], (int, float))
        assert body["uptime"] >= 0


# ---------------------------------------------------------------------------
# Agent CRUD
# ---------------------------------------------------------------------------

class TestAgents:
    def test_create_agent(self, client, admin_headers):
        resp = client.post(
            "/admin/agents",
            headers=admin_headers,
            json={
                "id": "test-agent-1",
                "name": "Test Agent",
                "url": "https://api.example.com/v1",
                "model": "gpt-4o-mini",
                "backend_type": "openai",
                "is_default": False,
            },
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["id"] == "test-agent-1"
        assert body["name"] == "Test Agent"
        assert body["model"] == "gpt-4o-mini"

    def test_list_agents(self, client, admin_headers):
        resp = client.get("/admin/agents", headers=admin_headers)
        assert resp.status_code == 200
        agents = resp.json()
        assert isinstance(agents, list)
        # Should have at least the seeded default + the one we just created
        ids = [a["id"] for a in agents]
        assert "test-agent-1" in ids

    def test_get_agent_detail(self, client, admin_headers):
        resp = client.get("/admin/agents/test-agent-1", headers=admin_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == "test-agent-1"

    def test_create_duplicate_agent_409(self, client, admin_headers):
        resp = client.post(
            "/admin/agents",
            headers=admin_headers,
            json={
                "id": "test-agent-1",
                "name": "Duplicate",
                "url": "https://api.example.com/v1",
                "model": "gpt-4o-mini",
            },
        )
        assert resp.status_code == 409

    def test_get_nonexistent_agent_404(self, client, admin_headers):
        resp = client.get("/admin/agents/no-such-agent", headers=admin_headers)
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# API Key CRUD
# ---------------------------------------------------------------------------

class TestAPIKeys:
    _created_key_id: str = ""
    _created_plaintext: str = ""

    def test_create_key(self, client, admin_headers):
        resp = client.post(
            "/admin/keys",
            headers=admin_headers,
            json={
                "name": "Test Key",
                "tier": "free",
                "rate_limit_per_minute": 30,
                "monthly_minutes": 120,
            },
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()

        # The plaintext key should be present and have the vr_live_ prefix
        assert "plaintext_key" in body
        assert body["plaintext_key"].startswith("vr_live_")
        TestAPIKeys._created_plaintext = body["plaintext_key"]

        # The key metadata should be nested under "key"
        key_meta = body["key"]
        assert key_meta["name"] == "Test Key"
        assert key_meta["tier"] == "free"
        assert key_meta["is_active"] is True
        TestAPIKeys._created_key_id = key_meta["id"]

    def test_list_keys(self, client, admin_headers):
        resp = client.get("/admin/keys", headers=admin_headers)
        assert resp.status_code == 200
        keys = resp.json()
        assert isinstance(keys, list)
        assert len(keys) >= 1
        ids = [k["id"] for k in keys]
        assert TestAPIKeys._created_key_id in ids

    def test_key_prefix_matches(self, client, admin_headers):
        resp = client.get("/admin/keys", headers=admin_headers)
        keys = resp.json()
        our_key = next(k for k in keys if k["id"] == TestAPIKeys._created_key_id)
        # The prefix should match the start of the plaintext key
        assert TestAPIKeys._created_plaintext.startswith(our_key["key_prefix"])

    def test_get_key_detail_with_usage(self, client, admin_headers):
        resp = client.get(
            f"/admin/keys/{TestAPIKeys._created_key_id}",
            headers=admin_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "key" in body
        assert "usage" in body
        assert body["usage"]["stt_seconds"] >= 0
        assert body["usage"]["tts_seconds"] >= 0


# ---------------------------------------------------------------------------
# System stats
# ---------------------------------------------------------------------------

class TestSystemStats:
    def test_get_stats(self, client, admin_headers):
        resp = client.get("/admin/stats", headers=admin_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert "active_sessions" in body
        assert "total_sessions" in body
        assert "total_api_keys" in body
        assert "total_agents" in body
        assert "uptime" in body
        assert body["uptime"] >= 0
        assert "stt_backend" in body
        assert "tts_backend" in body
        assert "vad_backend" in body

    def test_stats_counts_are_non_negative(self, client, admin_headers):
        resp = client.get("/admin/stats", headers=admin_headers)
        body = resp.json()
        assert body["active_sessions"] >= 0
        assert body["total_sessions"] >= 0
        assert body["total_api_keys"] >= 0
        assert body["total_agents"] >= 0


# ---------------------------------------------------------------------------
# Auth rejection (no admin key)
# ---------------------------------------------------------------------------

class TestAuthRejection:
    def test_agents_without_key_returns_401(self, client):
        resp = client.get("/admin/agents")
        assert resp.status_code == 401

    def test_keys_without_key_returns_401(self, client):
        resp = client.get("/admin/keys")
        assert resp.status_code == 401

    def test_stats_without_key_returns_401(self, client):
        resp = client.get("/admin/stats")
        assert resp.status_code == 401

    def test_create_agent_without_key_returns_401(self, client):
        resp = client.post(
            "/admin/agents",
            json={
                "id": "sneaky",
                "name": "Unauthorized",
                "url": "https://evil.com/v1",
                "model": "bad-model",
            },
        )
        assert resp.status_code == 401

    def test_create_key_without_key_returns_401(self, client):
        resp = client.post(
            "/admin/keys",
            json={"name": "Sneaky Key"},
        )
        assert resp.status_code == 401

    def test_wrong_key_returns_401(self, client):
        resp = client.get(
            "/admin/agents",
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 401

    def test_x_admin_key_header_works(self, client):
        resp = client.get(
            "/admin/agents",
            headers={"X-Admin-Key": MASTER_KEY},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Agent deletion and key deactivation
# ---------------------------------------------------------------------------

class TestCleanup:
    def test_delete_agent(self, client, admin_headers):
        resp = client.delete("/admin/agents/test-agent-1", headers=admin_headers)
        assert resp.status_code == 204

        # Confirm it is gone
        resp = client.get("/admin/agents/test-agent-1", headers=admin_headers)
        assert resp.status_code == 404

    def test_deactivate_key(self, client, admin_headers):
        key_id = TestAPIKeys._created_key_id
        resp = client.delete(f"/admin/keys/{key_id}", headers=admin_headers)
        assert resp.status_code == 204

        # Key should still show up in the list but be inactive
        resp = client.get("/admin/keys", headers=admin_headers)
        keys = resp.json()
        our_key = next((k for k in keys if k["id"] == key_id), None)
        assert our_key is not None
        assert our_key["is_active"] is False
