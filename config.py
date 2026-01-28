"""Configuration management for Voice AI Assistant."""

import fcntl
import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

CONFIG_DIR = Path.home() / ".config" / "voice-ai"
CONFIG_FILE = CONFIG_DIR / "config.json"
LOCK_FILE = CONFIG_DIR / "voice-ai.lock"

DEFAULT_CONFIG = {
    "hotkeys": {
        "hold_key": "alt_r",
        "toggle_modifier": "alt_l",
    },
    "stt": {
        "engine": "whisper",
        "model": "mlx-community/whisper-base-mlx",
    },
    "tts": {
        "engine": "kokoro",
        "voice": "af_heart",
    },
    "llm": {
        "endpoint": "http://localhost:1234/v1/chat/completions",
        "model": "local-model",
        "system_prompt": (
            "You are a voice assistant running on the user's Mac. "
            "Your responses are spoken aloud via text-to-speech, so keep them short, natural, and conversational — "
            "avoid markdown, bullet lists, code blocks, or other visual formatting. "
            "Prefer plain, spoken-style language. "
            "If a question is ambiguous, give a brief, practical answer rather than listing every possibility. "
            "Do not use emojis. "
            "When you don't know something, say so plainly."
        ),
    },
}

_lock_fd = None


def _deep_merge(defaults: dict, overrides: dict) -> dict:
    """Recursively merge overrides into defaults."""
    result = defaults.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config() -> dict:
    """Load config from disk, merging with defaults for any missing keys."""
    log.info("Loading config from %s", CONFIG_FILE)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                user_config = json.load(f)
            merged = _deep_merge(DEFAULT_CONFIG, user_config)
            log.info("Config loaded successfully — keys: %s", list(merged.keys()))
            log.debug("Full config: %s", json.dumps(merged, indent=2))
            return merged
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Failed to load config file, using defaults: %s", e)

    log.info("No existing config found — writing defaults to %s", CONFIG_FILE)
    save_config(DEFAULT_CONFIG)
    return DEFAULT_CONFIG.copy()


def save_config(config: dict) -> None:
    """Atomically write config to disk."""
    log.info("Saving config to %s", CONFIG_FILE)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = CONFIG_FILE.with_suffix(".tmp")
    try:
        with open(tmp_path, "w") as f:
            json.dump(config, f, indent=2)
        tmp_path.replace(CONFIG_FILE)
        log.debug("Config saved successfully")
    except OSError as e:
        log.error("Failed to save config: %s", e)
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def acquire_single_instance_lock() -> bool:
    """Acquire a file lock to ensure only one instance runs. Returns True if acquired."""
    global _lock_fd
    log.info("Acquiring single-instance lock at %s", LOCK_FILE)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        _lock_fd = open(LOCK_FILE, "w")
        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _lock_fd.write(str(os.getpid()))
        _lock_fd.flush()
        log.info("Lock acquired (PID %d)", os.getpid())
        return True
    except (OSError, IOError) as e:
        log.warning("Failed to acquire lock — another instance running? %s", e)
        if _lock_fd:
            _lock_fd.close()
            _lock_fd = None
        return False


def release_single_instance_lock() -> None:
    """Release the single instance lock."""
    global _lock_fd
    log.info("Releasing single-instance lock")
    if _lock_fd:
        try:
            fcntl.flock(_lock_fd, fcntl.LOCK_UN)
            _lock_fd.close()
            log.debug("Lock fd closed")
        except OSError as e:
            log.warning("Error releasing lock fd: %s", e)
        _lock_fd = None
    if LOCK_FILE.exists():
        try:
            LOCK_FILE.unlink()
            log.debug("Lock file removed")
        except OSError as e:
            log.warning("Error removing lock file: %s", e)
