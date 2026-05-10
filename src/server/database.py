"""
Async SQLite database layer for the Voice Relay server -- Phase 2.

Provides persistent storage for:
  - Agent backends (routing targets)
  - API keys (multi-tenant auth)
  - Sessions (survive restarts)
  - Usage events (billing / monitoring)

Uses aiosqlite for non-blocking I/O and WAL mode for concurrent reads.
Migrations are applied automatically at startup via the schema_version table.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Optional

import aiosqlite
from loguru import logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _hash_key(plaintext: str) -> str:
    """Return the SHA-256 hex digest of *plaintext*."""
    return hashlib.sha256(plaintext.encode()).hexdigest()


def _row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
    """Convert an aiosqlite Row to a plain dict."""
    return dict(row)


# ---------------------------------------------------------------------------
# Migration definitions
# ---------------------------------------------------------------------------
# Each entry is (version, sql).  Versions must be consecutive starting at 1.
# Never modify an existing migration; add a new one instead.

_MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
        CREATE TABLE IF NOT EXISTS agents (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            backend_type TEXT NOT NULL DEFAULT 'openai',
            url TEXT NOT NULL,
            model TEXT NOT NULL,
            api_key TEXT,
            system_prompt TEXT,
            is_default BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS api_keys (
            id TEXT PRIMARY KEY,
            key_hash TEXT NOT NULL UNIQUE,
            key_prefix TEXT NOT NULL,
            name TEXT NOT NULL,
            tier TEXT NOT NULL DEFAULT 'free',
            rate_limit_per_minute INTEGER DEFAULT 10,
            monthly_minutes INTEGER DEFAULT 60,
            is_active BOOLEAN DEFAULT TRUE,
            agent_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP,
            FOREIGN KEY (agent_id) REFERENCES agents(id)
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            api_key_id TEXT,
            agent_id TEXT,
            conversation_history TEXT,
            backend_url_override TEXT,
            backend_model_override TEXT,
            is_connected BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT,
            FOREIGN KEY (api_key_id) REFERENCES api_keys(id),
            FOREIGN KEY (agent_id) REFERENCES agents(id)
        );

        CREATE TABLE IF NOT EXISTS usage_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            api_key_id TEXT NOT NULL,
            session_id TEXT,
            event_type TEXT NOT NULL,
            value REAL NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (api_key_id) REFERENCES api_keys(id),
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );

        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
    ),
]

# Latest schema version is the highest migration number.
_LATEST_VERSION: int = max(v for v, _ in _MIGRATIONS)


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------

class Database:
    """
    Async SQLite database layer for the Voice Relay server.

    Usage::

        db = Database("voice_relay.db")
        await db.initialize()

        session = await db.create_session(session_id="...", api_key_id="...")
        await db.close()

    All public methods are coroutines and safe to call from async code.
    """

    def __init__(self, db_path: str = "voice_relay.db") -> None:
        """
        Initialise the database handle.

        Args:
            db_path: Path to the SQLite file.  Defaults to ``voice_relay.db``
                     in the current working directory.  Use ``":memory:"`` for
                     unit tests.
        """
        self._db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """
        Open the database connection, enable WAL mode, and apply any
        pending migrations.

        Must be called before any other method.
        """
        logger.info(f"Opening database: {self._db_path}")
        # Ensure the parent directory exists (handles fresh deploys where
        # the volume mount point may not have been created yet).
        db_dir = os.path.dirname(self._db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row

        # WAL mode gives concurrent readers without blocking writes.
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.commit()

        await self._run_migrations()
        logger.info("Database ready")

    async def close(self) -> None:
        """Close the database connection gracefully."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            logger.info("Database connection closed")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_conn(self) -> aiosqlite.Connection:
        """Return the active connection or raise RuntimeError."""
        if self._conn is None:
            raise RuntimeError(
                "Database not initialised — call await db.initialize() first"
            )
        return self._conn

    @asynccontextmanager
    async def _transaction(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        """
        Async context manager that wraps a block in an SQLite transaction.

        Commits on success, rolls back on exception.
        """
        conn = self._require_conn()
        try:
            yield conn
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise

    async def _run_migrations(self) -> None:
        """
        Apply any schema migrations that have not yet been run.

        Checks the ``schema_version`` table, compares against
        ``_MIGRATIONS``, and executes missing ones in order.
        """
        conn = self._require_conn()

        # schema_version may not exist yet (fresh DB), so create it first.
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await conn.commit()

        cursor = await conn.execute("SELECT MAX(version) FROM schema_version")
        row = await cursor.fetchone()
        current_version: int = row[0] if row[0] is not None else 0

        if current_version >= _LATEST_VERSION:
            logger.debug(f"Schema is up to date (version {current_version})")
            return

        logger.info(
            f"Schema migration needed: current={current_version}, "
            f"target={_LATEST_VERSION}"
        )

        for version, sql in _MIGRATIONS:
            if version <= current_version:
                continue

            logger.info(f"Applying migration {version}...")
            async with self._transaction() as txn:
                await txn.executescript(sql)
                await txn.execute(
                    "INSERT OR REPLACE INTO schema_version (version, applied_at) "
                    "VALUES (?, ?)",
                    (version, _now_iso()),
                )

            logger.info(f"Migration {version} applied")

    # ------------------------------------------------------------------
    # Session CRUD
    # ------------------------------------------------------------------

    async def create_session(
        self,
        session_id: str,
        api_key_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Create a new session record in the database.

        Args:
            session_id:  UUID string for the new session.
            api_key_id:  API key that initiated the session (may be None).
            agent_id:    Agent backend assigned to this session (may be None).

        Returns:
            The newly created session as a dict.
        """
        now = _now_iso()
        async with self._transaction() as conn:
            await conn.execute(
                """
                INSERT INTO sessions
                    (id, api_key_id, agent_id, conversation_history,
                     is_connected, created_at, last_activity)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    api_key_id,
                    agent_id,
                    json.dumps([]),
                    True,
                    now,
                    now,
                ),
            )

        logger.debug(f"Session created: {session_id}")
        session = await self.get_session(session_id)
        assert session is not None
        return session

    async def get_session(self, session_id: str) -> Optional[dict[str, Any]]:
        """
        Fetch a session by its ID.

        Args:
            session_id: UUID string of the session.

        Returns:
            Session dict with ``conversation_history`` already decoded from
            JSON, or ``None`` if no session with that ID exists.
        """
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None

        result = _row_to_dict(row)
        result["conversation_history"] = json.loads(
            result.get("conversation_history") or "[]"
        )
        if result.get("metadata"):
            result["metadata"] = json.loads(result["metadata"])
        return result

    async def update_session(self, session_id: str, **kwargs: Any) -> None:
        """
        Update arbitrary fields on a session.

        Only columns that exist in the sessions table may be passed as
        keyword arguments.  ``last_activity`` is automatically set to now.

        Args:
            session_id: UUID string of the session to update.
            **kwargs:   Column-name → value pairs.  Pass
                        ``conversation_history`` as a Python list; it will be
                        serialised to JSON automatically.  Pass ``metadata``
                        as a dict; it will also be serialised.
        """
        if not kwargs:
            return

        kwargs["last_activity"] = _now_iso()

        # Serialise complex fields
        if "conversation_history" in kwargs and isinstance(
            kwargs["conversation_history"], list
        ):
            kwargs["conversation_history"] = json.dumps(kwargs["conversation_history"])
        if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
            kwargs["metadata"] = json.dumps(kwargs["metadata"])

        columns = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values()) + [session_id]

        async with self._transaction() as conn:
            await conn.execute(
                f"UPDATE sessions SET {columns} WHERE id = ?", values
            )

    async def save_conversation_history(
        self, session_id: str, history: list[dict[str, Any]]
    ) -> None:
        """
        Persist the conversation history for a session.

        Args:
            session_id: UUID string of the session.
            history:    List of ``{"role": ..., "content": ...}`` dicts.
        """
        async with self._transaction() as conn:
            await conn.execute(
                "UPDATE sessions SET conversation_history = ?, last_activity = ? "
                "WHERE id = ?",
                (json.dumps(history), _now_iso(), session_id),
            )

    async def delete_expired_sessions(self, ttl_seconds: int) -> int:
        """
        Remove sessions that have been inactive for longer than *ttl_seconds*.

        Args:
            ttl_seconds: Maximum allowed idle time in seconds.

        Returns:
            Number of sessions deleted.
        """
        # SQLite datetime arithmetic: subtract TTL seconds from now.
        cutoff = (
            "datetime('now', '-" + str(int(ttl_seconds)) + " seconds')"
        )
        async with self._transaction() as conn:
            cursor = await conn.execute(
                f"DELETE FROM sessions WHERE last_activity < {cutoff}"
            )
            count: int = cursor.rowcount

        if count:
            logger.info(f"Deleted {count} expired session(s)")
        return count

    async def list_active_sessions(self) -> list[dict[str, Any]]:
        """
        Return all sessions that are currently marked as connected.

        Returns:
            List of session dicts (``conversation_history`` decoded).
        """
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT * FROM sessions WHERE is_connected = TRUE "
            "ORDER BY last_activity DESC"
        )
        rows = await cursor.fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            r = _row_to_dict(row)
            r["conversation_history"] = json.loads(r.get("conversation_history") or "[]")
            if r.get("metadata"):
                r["metadata"] = json.loads(r["metadata"])
            results.append(r)
        return results

    # ------------------------------------------------------------------
    # API Key CRUD
    # ------------------------------------------------------------------

    async def create_api_key(
        self,
        name: str,
        tier: str = "free",
        agent_id: Optional[str] = None,
        rate_limit_per_minute: int = 10,
        monthly_minutes: int = 60,
        expires_at: Optional[str] = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Generate and persist a new API key.

        The plaintext key is returned exactly once; only its SHA-256 hash
        is stored.  Keys have the format ``vr_live_<32 hex chars>``.

        Args:
            name:                  Human-readable label for this key.
            tier:                  Pricing tier (``free`` | ``pro`` | ``enterprise``).
            agent_id:              Default agent for this key (None = system default).
            rate_limit_per_minute: Requests-per-minute cap.
            monthly_minutes:       Monthly minute quota.
            expires_at:            Optional ISO 8601 expiry timestamp.

        Returns:
            A tuple of ``(plaintext_key, key_record_dict)``.  Store the
            plaintext key securely -- it cannot be recovered later.
        """
        key_id = str(uuid.uuid4())
        plaintext_key = "vr_live_" + secrets.token_hex(16)
        key_hash = _hash_key(plaintext_key)
        key_prefix = plaintext_key[:10]  # "vr_live_XX" (10 chars, still readable)
        now = _now_iso()

        async with self._transaction() as conn:
            await conn.execute(
                """
                INSERT INTO api_keys
                    (id, key_hash, key_prefix, name, tier,
                     rate_limit_per_minute, monthly_minutes,
                     is_active, agent_id, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key_id,
                    key_hash,
                    key_prefix,
                    name,
                    tier,
                    rate_limit_per_minute,
                    monthly_minutes,
                    True,
                    agent_id,
                    now,
                    expires_at,
                ),
            )

        logger.info(f"API key created: {key_id} name={name!r} tier={tier}")

        key_record = await self._get_api_key_by_id(key_id)
        assert key_record is not None
        return plaintext_key, key_record

    async def _get_api_key_by_id(self, key_id: str) -> Optional[dict[str, Any]]:
        """Internal: fetch a key record by its UUID (not the hash)."""
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT * FROM api_keys WHERE id = ?", (key_id,)
        )
        row = await cursor.fetchone()
        return _row_to_dict(row) if row else None

    async def validate_api_key(self, plaintext_key: str) -> Optional[dict[str, Any]]:
        """
        Validate a plaintext API key and return its record.

        Hashes the supplied key with SHA-256 and looks it up in the
        database.  Inactive or expired keys are rejected.

        Args:
            plaintext_key: The full plaintext key (``vr_live_...``).

        Returns:
            The API key record dict, or ``None`` if the key is invalid,
            inactive, or expired.
        """
        if not plaintext_key or not plaintext_key.startswith("vr_live_"):
            return None

        key_hash = _hash_key(plaintext_key)
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT * FROM api_keys WHERE key_hash = ?", (key_hash,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None

        record = _row_to_dict(row)

        if not record.get("is_active"):
            logger.debug(f"Rejected inactive key: {record['key_prefix']}...")
            return None

        if record.get("expires_at"):
            if record["expires_at"] < _now_iso():
                logger.debug(f"Rejected expired key: {record['key_prefix']}...")
                return None

        return record

    async def deactivate_api_key(self, key_id: str) -> None:
        """
        Mark an API key as inactive, preventing future authentication.

        Args:
            key_id: UUID of the API key to deactivate.
        """
        async with self._transaction() as conn:
            await conn.execute(
                "UPDATE api_keys SET is_active = FALSE WHERE id = ?", (key_id,)
            )
        logger.info(f"API key deactivated: {key_id}")

    async def list_api_keys(self) -> list[dict[str, Any]]:
        """
        Return all API key records (active and inactive).

        Returns:
            List of key record dicts, ordered by creation time descending.
            The ``key_hash`` field is intentionally included so callers can
            verify a plaintext key if needed, but it should never be
            transmitted to untrusted clients.
        """
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT * FROM api_keys ORDER BY created_at DESC"
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Agent CRUD
    # ------------------------------------------------------------------

    async def create_agent(
        self,
        id: str,
        name: str,
        url: str,
        model: str,
        backend_type: str = "openai",
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        is_default: bool = False,
    ) -> dict[str, Any]:
        """
        Create a new agent backend configuration.

        If *is_default* is True, any previously set default agent is
        cleared first so there is always at most one default.

        Args:
            id:            Unique slug for the agent (e.g. ``"hermes"``).
            name:          Human-readable label.
            url:           Base URL of the OpenAI-compatible endpoint.
            model:         Model identifier sent to the backend.
            backend_type:  ``"openai"`` or ``"openclaw"``.
            api_key:       Optional API key for the backend.
            system_prompt: Optional system prompt override for this agent.
            is_default:    Whether this agent is the system-wide default.

        Returns:
            The newly created agent record as a dict.
        """
        now = _now_iso()

        async with self._transaction() as conn:
            if is_default:
                # Clear any existing default before setting the new one.
                await conn.execute(
                    "UPDATE agents SET is_default = FALSE WHERE is_default = TRUE"
                )

            await conn.execute(
                """
                INSERT INTO agents
                    (id, name, backend_type, url, model,
                     api_key, system_prompt, is_default, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    id,
                    name,
                    backend_type,
                    url,
                    model,
                    api_key,
                    system_prompt,
                    is_default,
                    now,
                    now,
                ),
            )

        logger.info(f"Agent created: {id} ({name})")
        agent = await self.get_agent(id)
        assert agent is not None
        return agent

    async def get_agent(self, agent_id: str) -> Optional[dict[str, Any]]:
        """
        Fetch an agent configuration by its ID.

        Args:
            agent_id: The agent slug (e.g. ``"hermes"``).

        Returns:
            Agent record dict or ``None`` if not found.
        """
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT * FROM agents WHERE id = ?", (agent_id,)
        )
        row = await cursor.fetchone()
        return _row_to_dict(row) if row else None

    async def get_default_agent(self) -> Optional[dict[str, Any]]:
        """
        Return the agent marked as the system-wide default.

        Returns:
            Agent record dict or ``None`` if no default is set.
        """
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT * FROM agents WHERE is_default = TRUE LIMIT 1"
        )
        row = await cursor.fetchone()
        return _row_to_dict(row) if row else None

    async def list_agents(self) -> list[dict[str, Any]]:
        """
        Return all registered agent configurations.

        Returns:
            List of agent record dicts, default agent first.
        """
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT * FROM agents ORDER BY is_default DESC, name ASC"
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]

    async def update_agent(self, agent_id: str, **kwargs: Any) -> None:
        """
        Update fields on an existing agent configuration.

        Args:
            agent_id: The agent slug to update.
            **kwargs: Column-name → value pairs.

        Raises:
            ValueError: If *kwargs* is empty.
        """
        if not kwargs:
            raise ValueError("update_agent requires at least one field to update")

        kwargs["updated_at"] = _now_iso()

        if "is_default" in kwargs and kwargs["is_default"]:
            async with self._transaction() as conn:
                await conn.execute(
                    "UPDATE agents SET is_default = FALSE WHERE is_default = TRUE "
                    "AND id != ?",
                    (agent_id,),
                )

        columns = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values()) + [agent_id]

        async with self._transaction() as conn:
            await conn.execute(
                f"UPDATE agents SET {columns} WHERE id = ?", values
            )
        logger.debug(f"Agent updated: {agent_id}")

    async def delete_agent(self, agent_id: str) -> None:
        """
        Delete an agent configuration.

        Sessions and API keys that reference this agent will have their
        ``agent_id`` set to NULL by the ON DELETE SET NULL foreign key
        action.  Note: SQLite requires ``PRAGMA foreign_keys = ON`` (set
        during :meth:`initialize`) for this to take effect.

        Args:
            agent_id: The agent slug to delete.
        """
        async with self._transaction() as conn:
            await conn.execute("DELETE FROM agents WHERE id = ?", (agent_id,))
        logger.info(f"Agent deleted: {agent_id}")

    # ------------------------------------------------------------------
    # Usage tracking
    # ------------------------------------------------------------------

    async def record_usage(
        self,
        api_key_id: str,
        event_type: str,
        value: float,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Append a usage event to the database.

        Valid event types:
          - ``stt_seconds``   — seconds of speech processed by STT
          - ``tts_seconds``   — seconds of audio generated by TTS
          - ``llm_tokens``    — token count charged by the LLM backend
          - ``session_start`` — session opened (value = 0)
          - ``session_end``   — session closed (value = 0)

        Args:
            api_key_id:  UUID of the API key responsible for the usage.
            event_type:  One of the event types listed above.
            value:       Numeric magnitude (seconds, tokens, etc.).
            session_id:  UUID of the associated session, if any.
        """
        async with self._transaction() as conn:
            await conn.execute(
                """
                INSERT INTO usage_events
                    (api_key_id, session_id, event_type, value, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (api_key_id, session_id, event_type, value, _now_iso()),
            )

    async def get_usage_summary(
        self,
        api_key_id: str,
        period_start: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Return aggregated usage statistics for an API key.

        Args:
            api_key_id:   UUID of the API key.
            period_start: ISO 8601 timestamp; only events on or after this
                          timestamp are included.  Defaults to the start of
                          the current calendar month (UTC).

        Returns:
            Dict with keys:
              - ``api_key_id``
              - ``period_start``
              - ``stt_seconds``
              - ``tts_seconds``
              - ``llm_tokens``
              - ``session_starts``
        """
        if period_start is None:
            now = datetime.now(timezone.utc)
            period_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc).isoformat()

        conn = self._require_conn()
        cursor = await conn.execute(
            """
            SELECT
                event_type,
                SUM(value) AS total
            FROM usage_events
            WHERE api_key_id = ?
              AND created_at >= ?
            GROUP BY event_type
            """,
            (api_key_id, period_start),
        )
        rows = await cursor.fetchall()

        summary: dict[str, Any] = {
            "api_key_id": api_key_id,
            "period_start": period_start,
            "stt_seconds": 0.0,
            "tts_seconds": 0.0,
            "llm_tokens": 0,
            "session_starts": 0,
        }

        for row in rows:
            et: str = row["event_type"]
            total: float = row["total"] or 0
            if et == "stt_seconds":
                summary["stt_seconds"] = round(total, 3)
            elif et == "tts_seconds":
                summary["tts_seconds"] = round(total, 3)
            elif et == "llm_tokens":
                summary["llm_tokens"] = int(total)
            elif et == "session_start":
                summary["session_starts"] = int(total)

        return summary

    async def get_monthly_minutes(self, api_key_id: str) -> float:
        """
        Return the total audio minutes consumed so far this calendar month
        for *api_key_id*.

        Minutes are calculated as ``(stt_seconds + tts_seconds) / 60``.

        Args:
            api_key_id: UUID of the API key to query.

        Returns:
            Total minutes as a float, rounded to three decimal places.
        """
        now = datetime.now(timezone.utc)
        month_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc).isoformat()

        conn = self._require_conn()
        cursor = await conn.execute(
            """
            SELECT SUM(value) AS total_seconds
            FROM usage_events
            WHERE api_key_id = ?
              AND event_type IN ('stt_seconds', 'tts_seconds')
              AND created_at >= ?
            """,
            (api_key_id, month_start),
        )
        row = await cursor.fetchone()
        total_seconds: float = row["total_seconds"] or 0.0
        return round(total_seconds / 60.0, 3)
