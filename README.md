# VoiceAI Assistant

A fully offline macOS menu bar voice assistant. Speak into your mic, get a spoken AI response — all running locally on Apple Silicon.

**Pipeline:** Record → Transcribe (STT) → LLM → Synthesize (TTS) → Play

## Requirements

- macOS with Apple Silicon
- Python 3.10+
- A local LLM server (e.g., [LM Studio](https://lmstudio.ai)) running at `http://localhost:1234`

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

The app appears in your menu bar. On first run, STT/TTS models are downloaded from HuggingFace automatically.

## Usage

- **Hold Right Option** — hold to record, release to process
- **Left Option + Right Option** — toggle recording on/off
- All hotkeys, engines, voices, and LLM settings are configurable from the menu bar

## Supported Engines

| Component | Engine | Details |
|-----------|--------|---------|
| STT | MLX Whisper | Tiny, Base, Small, Medium, Large v3 |
| STT | Parakeet MLX | 0.6B v2 |
| TTS | Kokoro | 11 voice presets via mlx-audio |
| TTS | Supertonic | ONNX-based |
| LLM | Any OpenAI-compatible API | Tested with LM Studio |

## Configuration

Settings are stored at `~/.config/voice-ai/config.json` and can be changed from the menu bar UI.
