# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VoiceAI Assistant — a macOS menu bar app providing a fully offline voice assistant pipeline: speak → transcribe (STT) → LLM → synthesize (TTS) → hear response. Built in Python, optimized for Apple Silicon via MLX.

## Running the App

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run
python3 main.py
```

Requires a local LLM server (e.g., LM Studio) at `http://localhost:1234/v1/chat/completions`.

No test suite, linter config, or build system exists yet.

## Architecture

Seven modules, each independently testable:

- **main.py** — rumps menu bar app, orchestrator, state machine (idle → recording → transcribing → thinking → speaking)
- **config.py** — Config load/save at `~/.config/voice-ai/config.json`, single-instance file lock via `fcntl`
- **audio.py** — `AudioCapture` (16kHz mono float32 via sounddevice callbacks) and `AudioPlayer` (blocking playback with stop-flag interruption, automatic sample rate resampling via scipy)
- **hotkeys.py** — Global keyboard listener via pynput with hold-to-record and modifier+key toggle modes, runtime rebinding
- **stt.py** — Pluggable STT engines: MLX Whisper (5 model sizes) and Parakeet MLX. Lazy model loading with mutex
- **tts.py** — Pluggable TTS engines: Kokoro (via mlx-audio, 3-tier fallback loading strategy) and Supertonic (ONNX). 24kHz output
- **llm.py** — OpenAI-compatible API client with bounded conversation history (50 exchanges max), 120s timeout, health check via `/models`

## Key Design Patterns

**Pipeline execution**: Each recording triggers a background daemon thread running transcribe → LLM → TTS → play, guarded by a mutex to prevent concurrent pipeline runs.

**Interruption**: Pressing the hotkey during TTS playback immediately stops audio (via `threading.Event` stop flag) and starts a new recording.

**UI updates**: Menu bar state changes are queued via `queue.Queue` and polled every 0.1s by a rumps timer — required because rumps callbacks must run on the main thread.

**Model caching**: STT and TTS models are loaded once and kept in process memory. Preloading happens asynchronously on startup.

**TTS fallback chain** (Kokoro): Tries `KokoroPipeline` → `load_model` generic API → standalone `kokoro.KPipeline`.

## Config Location

`~/.config/voice-ai/config.json` — holds hotkey bindings, STT/TTS engine+model selections, LLM endpoint/model/system prompt.
