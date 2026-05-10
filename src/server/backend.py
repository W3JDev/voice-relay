"""
AI Backend module for the Phase 1 Voice Relay.

Connects to OpenAI-compatible endpoints (OpenAI, OpenClaw, local LLMs).
Conversation history is managed externally (per-session); this module
only builds the prompt from the supplied history and streams the response.
"""

import asyncio
from typing import Optional, List, Dict, AsyncGenerator, Tuple

from loguru import logger


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful voice assistant. "
    "Keep your responses concise and conversational -- one to two sentences "
    "unless the user asks for more detail. "
    "Avoid markdown, bullet lists, or other formatting that doesn't translate "
    "well to speech. Use natural spoken language."
)

# How many recent turns (user + assistant pairs) to include in the context
# window sent to the model.
MAX_HISTORY_TURNS = 10


class AIBackend:
    """AI backend for processing user messages.

    Unlike the upstream version, this class does **not** hold any
    conversation history.  History is passed in by the caller so that
    each WebSocket session can maintain its own independent context.
    """

    def __init__(
        self,
        backend_type: str = "openai",
        url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.backend_type = backend_type
        self.url = url
        self.model = model
        self.api_key = api_key
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._client = None
        self._setup_client()

    # ------------------------------------------------------------------
    # Client setup
    # ------------------------------------------------------------------
    def _setup_client(self) -> None:
        """Initialise the AsyncOpenAI client."""
        if self.backend_type in ("openai", "openclaw"):
            try:
                from openai import AsyncOpenAI

                # Use a dummy key if none provided so the client can be constructed.
                # Actual API calls will fail gracefully in chat()/chat_stream().
                key = self.api_key or "not-configured"
                self._client = AsyncOpenAI(
                    api_key=key,
                    base_url=self.url if self.url != "https://api.openai.com/v1" else None,
                )
                if self.api_key:
                    logger.info(
                        f"AI backend ready  (type={self.backend_type}, "
                        f"model={self.model}, url={self.url})"
                    )
                else:
                    logger.warning(
                        f"AI backend created without API key  (type={self.backend_type}). "
                        "Responses will echo back user input until a key is configured."
                    )
                    self._client = None  # Fall back to echo mode
            except ImportError:
                logger.error("openai package not installed -- AI backend unavailable")
            except Exception as exc:
                logger.error(f"Failed to initialise AI backend: {exc}")
                self._client = None
        else:
            logger.warning(f"Unknown backend type: {self.backend_type}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def chat(
        self,
        user_message: str,
        history: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
    ) -> Tuple[str, List[Dict]]:
        """Send a message and return the full response.

        Args:
            user_message:  The user's transcribed speech.
            history:       Conversation history (list of ``{"role": ..., "content": ...}``
                           dicts).  A *new* list is returned with the exchange appended --
                           the original is **not** mutated.
            system_prompt: Override the instance-level system prompt for this call.

        Returns:
            A tuple of ``(response_text, updated_history)``.
        """
        history = list(history) if history else []
        history.append({"role": "user", "content": user_message})

        messages = self._build_messages(history, system_prompt)

        if self._client is None:
            reply = f"I heard you say: {user_message}"
            history.append({"role": "assistant", "content": reply})
            return reply, history

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500,
                temperature=0.7,
            )
            reply = response.choices[0].message.content or ""
        except Exception as exc:
            logger.error(f"OpenAI API error: {exc}")
            reply = "Sorry, I had trouble processing that. Could you try again?"

        history.append({"role": "assistant", "content": reply})
        return reply, history

    async def chat_stream(
        self,
        user_message: str,
        history: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncGenerator[Tuple[str, Optional[str]], None]:
        """Stream a response, yielding text chunks as they arrive.

        Args:
            user_message:  The user's transcribed speech.
            history:       Conversation history -- same semantics as ``chat()``.
                           Not mutated; the full response is delivered with the
                           final yield (see below).
            system_prompt: Override the instance-level system prompt for this call.
            cancel_event:  If this ``asyncio.Event`` is set while streaming, the
                           generator stops early (supports barge-in / interruption).

        Yields:
            ``(chunk, None)`` for each incremental text fragment.
            After the last real chunk, yields ``("", full_response)`` so the
            caller can append the complete assistant message to its history
            without having to accumulate chunks itself.

            If the stream is cancelled via *cancel_event*, the final yield
            contains whatever partial response was received up to that point.
        """
        history = list(history) if history else []
        history.append({"role": "user", "content": user_message})

        messages = self._build_messages(history, system_prompt)

        if self._client is None:
            fallback = f"I heard you say: {user_message}"
            yield (fallback, None)
            yield ("", fallback)
            return

        full_response = ""

        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500,
                temperature=0.7,
                stream=True,
            )

            async for chunk in stream:
                # Check for cancellation (barge-in)
                if cancel_event is not None and cancel_event.is_set():
                    logger.debug(
                        "chat_stream cancelled (barge-in) after "
                        f"{len(full_response)} chars"
                    )
                    break

                delta = chunk.choices[0].delta.content if chunk.choices[0].delta.content else None
                if delta:
                    full_response += delta
                    yield (delta, None)

        except Exception as exc:
            logger.error(f"OpenAI streaming error: {exc}")
            if not full_response:
                err_msg = "Sorry, I had trouble processing that."
                full_response = err_msg
                yield (err_msg, None)

        # Final sentinel: deliver the accumulated response
        yield ("", full_response)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_messages(
        self,
        history: List[Dict],
        system_prompt: Optional[str] = None,
    ) -> List[Dict]:
        """Construct the messages list for the API call.

        Includes the system prompt followed by the most recent turns from
        *history* (capped at ``MAX_HISTORY_TURNS`` messages).
        """
        prompt = system_prompt or self.system_prompt
        messages: List[Dict] = [{"role": "system", "content": prompt}]
        messages.extend(history[-MAX_HISTORY_TURNS:])
        return messages
