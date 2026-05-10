"""
Multi-agent routing for the Voice Relay.

Manages a registry of AI backends (agents) and routes sessions to the
appropriate one based on:
  1. Per-session config override (client sends {"type":"config","agent":{...}})
  2. API key's default agent assignment
  3. System default agent (is_default=True in DB)
  4. Fallback to env-var based backend (Phase 1 compatibility)

Usage::

    router = AgentRouter(db)
    await router.seed_defaults()

    # Resolve which backend to use for a session
    backend = await router.resolve_backend(session, api_key_record=key)

    # Invalidate cache after an admin config change
    router.invalidate_cache(agent_id="hermes")
"""

from __future__ import annotations

import os
from typing import Dict, Optional

from loguru import logger

from .backend import AIBackend
from .database import Database


class AgentRouter:
    """Routes voice sessions to the correct AI backend.

    Maintains an in-process cache of :class:`~backend.AIBackend` instances
    keyed by agent ID so that repeated requests reuse already-initialised
    clients rather than constructing new ones on every turn.

    The cache is invalidated explicitly (via :meth:`invalidate_cache`) when
    the admin API modifies an agent record, ensuring the next request picks
    up the updated configuration.
    """

    def __init__(self, db: Database) -> None:
        """
        Initialise the router.

        Args:
            db: Initialised :class:`~database.Database` instance used to look
                up agent configurations.
        """
        self._db = db
        self._backend_cache: Dict[str, AIBackend] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def resolve_backend(
        self,
        session,
        api_key_record: Optional[dict] = None,
    ) -> AIBackend:
        """Resolve which AI backend to use for a session.

        The resolution priority is:

        1. **Session-level override** — the client sent a ``{"type":"config",
           "agent": {...}}`` message that set ``backend_url_override`` and/or
           ``backend_model_override`` on the session.
        2. **API key assignment** — the API key record has a non-null
           ``agent_id`` column pointing to a registered agent.
        3. **System default agent** — an agent row with ``is_default=TRUE``
           in the database.
        4. **Env-var fallback** — construct an :class:`~backend.AIBackend`
           directly from the Phase 1 environment variables
           (``OPENCLAW_BACKEND_URL``, ``OPENCLAW_BACKEND_MODEL``,
           ``OPENCLAW_OPENAI_API_KEY`` / ``OPENAI_API_KEY``).

        Args:
            session: A :class:`~main.SessionState` instance.  The method reads
                     ``backend_url_override`` and ``backend_model_override``
                     from it to detect priority-1 overrides.
            api_key_record: The API key dict returned by
                            :meth:`~database.Database.validate_api_key`, or
                            ``None`` if the connection is unauthenticated.

        Returns:
            A ready-to-use :class:`~backend.AIBackend` instance.
        """
        # ------------------------------------------------------------------
        # Priority 1: session-level override (set via "config" message)
        # ------------------------------------------------------------------
        url_override = getattr(session, "backend_url_override", None)
        model_override = getattr(session, "backend_model_override", None)

        if url_override or model_override:
            logger.debug(
                f"Session {session.session_id}: using per-session backend override "
                f"(url={url_override}, model={model_override})"
            )
            return self._build_env_backend(
                url_override=url_override,
                model_override=model_override,
            )

        # ------------------------------------------------------------------
        # Priority 2: API key's assigned agent
        # ------------------------------------------------------------------
        if api_key_record and api_key_record.get("agent_id"):
            agent_id: str = api_key_record["agent_id"]
            logger.debug(
                f"Session {session.session_id}: using agent from API key: {agent_id}"
            )
            try:
                return await self.get_or_create_backend(agent_id)
            except LookupError:
                logger.warning(
                    f"Agent '{agent_id}' assigned to API key not found in DB; "
                    "falling through to system default"
                )

        # ------------------------------------------------------------------
        # Priority 3: System default agent (is_default=TRUE in DB)
        # ------------------------------------------------------------------
        default_agent = await self._db.get_default_agent()
        if default_agent:
            logger.debug(
                f"Session {session.session_id}: using system default agent: "
                f"{default_agent['id']}"
            )
            return await self.get_or_create_backend(default_agent["id"])

        # ------------------------------------------------------------------
        # Priority 4: Env-var fallback (Phase 1 compatibility)
        # ------------------------------------------------------------------
        logger.debug(
            f"Session {session.session_id}: no agent configured in DB; "
            "falling back to env-var backend"
        )
        return self._build_env_backend()

    async def get_or_create_backend(self, agent_id: str) -> AIBackend:
        """Return a cached backend or build one from the DB configuration.

        The backend is cached for the lifetime of the process (or until
        :meth:`invalidate_cache` is called for this agent).

        Args:
            agent_id: The slug of the agent to look up (e.g. ``"hermes"``).

        Returns:
            A ready-to-use :class:`~backend.AIBackend` instance.

        Raises:
            LookupError: If *agent_id* is not found in the database.
        """
        if agent_id in self._backend_cache:
            return self._backend_cache[agent_id]

        agent = await self._db.get_agent(agent_id)
        if agent is None:
            raise LookupError(f"Agent not found in database: {agent_id!r}")

        backend = AIBackend(
            backend_type=agent.get("backend_type", "openai"),
            url=agent["url"],
            model=agent["model"],
            api_key=agent.get("api_key"),
            system_prompt=agent.get("system_prompt"),
        )

        self._backend_cache[agent_id] = backend
        logger.debug(f"Backend cached for agent: {agent_id}")
        return backend

    async def seed_defaults(self) -> None:
        """Seed the default agent from environment variables if no agents exist.

        Reads the following environment variables (all optional):

        - ``OPENCLAW_BACKEND_URL`` — base URL of the LLM endpoint
          (defaults to ``https://api.openai.com/v1``)
        - ``OPENCLAW_BACKEND_MODEL`` — model identifier
          (defaults to ``gpt-4o-mini``)
        - ``OPENCLAW_OPENAI_API_KEY`` or ``OPENAI_API_KEY`` — API key for
          the backend
        - ``OPENCLAW_BACKEND_TYPE`` — ``"openai"`` or ``"openclaw"``
          (defaults to ``"openai"``)
        - ``OPENCLAW_GATEWAY_URL`` / ``OPENCLAW_GATEWAY_TOKEN`` — if set,
          these take precedence and configure an OpenClaw gateway backend

        This method is a no-op if at least one agent already exists in the
        database, so it is safe to call on every startup.
        """
        existing = await self._db.list_agents()
        if existing:
            logger.debug(
                f"seed_defaults: {len(existing)} agent(s) already in DB; skipping"
            )
            return

        # Prefer the OpenClaw gateway if both vars are set (matches _create_backend
        # in main.py).
        gateway_url = os.environ.get("OPENCLAW_GATEWAY_URL")
        gateway_token = os.environ.get("OPENCLAW_GATEWAY_TOKEN")

        if gateway_url and gateway_token:
            url = f"{gateway_url.rstrip('/')}/v1"
            model = "openclaw:voice"
            api_key_val = gateway_token
            backend_type = "openai"
            name = "OpenClaw Gateway (seeded)"
            logger.info(f"seed_defaults: seeding OpenClaw gateway agent from env ({url})")
        else:
            url = (
                os.environ.get("OPENCLAW_BACKEND_URL")
                or "https://api.openai.com/v1"
            )
            model = (
                os.environ.get("OPENCLAW_BACKEND_MODEL")
                or "gpt-4o-mini"
            )
            api_key_val = (
                os.environ.get("OPENCLAW_OPENAI_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            )
            backend_type = os.environ.get("OPENCLAW_BACKEND_TYPE", "openai")
            name = "Default (seeded from env)"
            logger.info(f"seed_defaults: seeding default agent from env (url={url}, model={model})")

        agent_id = "default"
        try:
            await self._db.create_agent(
                id=agent_id,
                name=name,
                url=url,
                model=model,
                backend_type=backend_type,
                api_key=api_key_val,
                is_default=True,
            )
            logger.info(f"seed_defaults: created agent '{agent_id}' as system default")
        except Exception as exc:
            logger.error(f"seed_defaults: failed to create default agent: {exc}")

    def invalidate_cache(self, agent_id: Optional[str] = None) -> None:
        """Clear cached backends so the next request reads fresh DB config.

        Call this after the admin API creates, updates, or deletes an agent
        so that stale backend objects are not reused.

        Args:
            agent_id: If provided, only evict the cache entry for this agent.
                      If ``None``, flush the entire cache.
        """
        if agent_id is None:
            count = len(self._backend_cache)
            self._backend_cache.clear()
            logger.debug(f"Backend cache fully invalidated ({count} entries cleared)")
        elif agent_id in self._backend_cache:
            del self._backend_cache[agent_id]
            logger.debug(f"Backend cache entry evicted: {agent_id}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_env_backend(
        self,
        url_override: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> AIBackend:
        """Construct a backend from environment variables plus optional overrides.

        Mirrors the logic in ``main._create_backend()`` so Phase 1 env-var
        configs continue to work when no DB agents are configured.

        Args:
            url_override:   If provided, replaces the env-var URL.
            model_override: If provided, replaces the env-var model.

        Returns:
            A freshly constructed :class:`~backend.AIBackend` (not cached,
            because session-level overrides are unique per session).
        """
        gateway_url = os.environ.get("OPENCLAW_GATEWAY_URL")
        gateway_token = os.environ.get("OPENCLAW_GATEWAY_TOKEN")

        if gateway_url and gateway_token and not url_override:
            return AIBackend(
                backend_type="openai",
                url=f"{gateway_url.rstrip('/')}/v1",
                model=model_override or "openclaw:voice",
                api_key=gateway_token,
            )

        return AIBackend(
            backend_type=os.environ.get("OPENCLAW_BACKEND_TYPE", "openai"),
            url=url_override or os.environ.get("OPENCLAW_BACKEND_URL", "https://api.openai.com/v1"),
            model=model_override or os.environ.get("OPENCLAW_BACKEND_MODEL", "gpt-4o-mini"),
            api_key=(
                os.environ.get("OPENCLAW_OPENAI_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            ),
        )
