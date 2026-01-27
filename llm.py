"""LLM client for Voice AI Assistant (OpenAI-compatible API)."""

import logging
import time

import requests

log = logging.getLogger(__name__)

MAX_HISTORY = 50  # Max conversation exchanges to keep


class LLMClient:
    """Client for OpenAI-compatible LLM APIs (e.g., LM Studio)."""

    def __init__(self, endpoint: str, model: str, system_prompt: str):
        self.endpoint = endpoint
        self.model = model
        self.system_prompt = system_prompt
        self.conversation_history: list[dict] = []
        log.info("LLMClient initialized — endpoint=%s, model=%s", endpoint, model)
        log.debug("LLMClient: system_prompt=%r", system_prompt[:100])

    def chat(self, user_message: str) -> str:
        """Send a message and return the assistant's response."""
        log.info("LLMClient.chat() — user_message=%r (%d chars)",
                 user_message[:100], len(user_message))

        self.conversation_history.append({"role": "user", "content": user_message})

        # Trim history to avoid unbounded growth
        if len(self.conversation_history) > MAX_HISTORY * 2:
            old_len = len(self.conversation_history)
            self.conversation_history = self.conversation_history[-(MAX_HISTORY * 2) :]
            log.info("LLMClient: Trimmed history from %d to %d messages", old_len, len(self.conversation_history))

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)
        log.info("LLMClient: Sending %d messages to %s (model=%s)",
                 len(messages), self.endpoint, self.model)

        t0 = time.time()
        try:
            response = requests.post(
                self.endpoint,
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                },
                timeout=120,
            )
            elapsed = time.time() - t0
            log.info("LLMClient: Response received — status=%d, elapsed=%.2fs",
                     response.status_code, elapsed)
            response.raise_for_status()
        except requests.RequestException as e:
            elapsed = time.time() - t0
            log.error("LLMClient: Request failed after %.2fs: %s", elapsed, e, exc_info=True)
            raise

        data = response.json()
        log.debug("LLMClient: Response JSON keys=%s", list(data.keys()))

        assistant_text = data["choices"][0]["message"]["content"]
        self.conversation_history.append(
            {"role": "assistant", "content": assistant_text}
        )
        log.info("LLMClient: Assistant response=%r (%d chars)",
                 assistant_text[:100], len(assistant_text))
        log.info("LLMClient: Conversation history now has %d messages",
                 len(self.conversation_history))
        return assistant_text

    def clear_history(self):
        """Clear conversation history."""
        count = len(self.conversation_history)
        self.conversation_history.clear()
        log.info("LLMClient: Cleared %d messages from history", count)

    def is_available(self) -> bool:
        """Check if the LLM endpoint is reachable."""
        base_url = self.endpoint.rsplit("/", 1)[0]
        check_url = f"{base_url}/models"
        log.info("LLMClient.is_available() — checking %s", check_url)
        try:
            resp = requests.get(check_url, timeout=2)
            log.info("LLMClient: Health check status=%d — available", resp.status_code)
            return True
        except Exception as e:
            log.warning("LLMClient: Health check failed — not available: %s", e)
            return False
