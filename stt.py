"""Speech-to-text engines for Voice AI Assistant."""

import logging
import threading
import time

log = logging.getLogger(__name__)

_loaded_model = None
_loaded_engine = None
_loaded_model_repo = None
_model_operation_in_progress = False
_model_lock = threading.Lock()

AVAILABLE_MODELS = {
    "whisper": {
        "Tiny (~75 MB)": "mlx-community/whisper-tiny-mlx",
        "Base (~140 MB)": "mlx-community/whisper-base-mlx",
        "Small (~460 MB)": "mlx-community/whisper-small-mlx",
        "Medium (~1.5 GB)": "mlx-community/whisper-medium-mlx",
        "Large v3 (~3 GB)": "mlx-community/whisper-large-v3-mlx",
    },
    "parakeet": {
        "0.6B v2 (~2.5 GB)": "mlx-community/parakeet-tdt-0.6b-v2",
    },
}


def preload_model(model_repo: str, engine: str, on_status=None, on_complete=None):
    """Preload an STT model in a background thread."""
    global _model_operation_in_progress

    log.info("preload_model() called — engine=%s, model=%s", engine, model_repo)

    if _model_operation_in_progress:
        log.warning("preload_model: Another model operation is already in progress, skipping")
        if on_complete:
            on_complete(False)
        return

    def _load():
        global _loaded_model, _loaded_engine, _loaded_model_repo, _model_operation_in_progress
        _model_operation_in_progress = True
        t0 = time.time()
        try:
            if on_status:
                on_status(f"Loading {engine} model...")
            log.info("preload_model: Loading %s model from %s...", engine, model_repo)

            if engine == "whisper":
                _load_whisper_model(model_repo)
            elif engine == "parakeet":
                _load_parakeet_model(model_repo)
            else:
                raise ValueError(f"Unknown STT engine: {engine}")

            _loaded_engine = engine
            _loaded_model_repo = model_repo

            elapsed = time.time() - t0
            log.info("preload_model: %s model loaded in %.2fs", engine, elapsed)
            if on_status:
                on_status("Model loaded")
            if on_complete:
                on_complete(True)
        except Exception as e:
            elapsed = time.time() - t0
            log.error("preload_model: Failed to load %s model after %.2fs: %s",
                      engine, elapsed, e, exc_info=True)
            if on_status:
                on_status(f"Failed to load model: {e}")
            if on_complete:
                on_complete(False)
        finally:
            _model_operation_in_progress = False

    threading.Thread(target=_load, daemon=True).start()


def _load_whisper_model(model_repo: str):
    """Pre-download/cache the whisper model."""
    global _loaded_model
    log.info("_load_whisper_model: Importing mlx_whisper...")
    import mlx_whisper  # noqa: F401
    log.info("_load_whisper_model: mlx_whisper imported, storing repo=%s", model_repo)
    # mlx_whisper handles model caching internally on first transcribe call
    _loaded_model = model_repo


def _load_parakeet_model(model_repo: str):
    """Load and cache the parakeet model."""
    global _loaded_model
    log.info("_load_parakeet_model: Importing parakeet_mlx...")
    from parakeet_mlx import from_pretrained
    log.info("_load_parakeet_model: Loading model from %s...", model_repo)
    _loaded_model = from_pretrained(model_repo)
    log.info("_load_parakeet_model: Model loaded successfully")


def transcribe(audio_path: str, engine: str, model_repo: str) -> str:
    """Transcribe audio file to text."""
    log.info("transcribe() called — engine=%s, model=%s, audio=%s", engine, model_repo, audio_path)
    t0 = time.time()

    with _model_lock:
        if engine == "whisper":
            text = _transcribe_whisper(audio_path, model_repo)
        elif engine == "parakeet":
            text = _transcribe_parakeet(audio_path, model_repo)
        else:
            raise ValueError(f"Unknown STT engine: {engine}")

    elapsed = time.time() - t0
    log.info("transcribe: Completed in %.2fs — result: %r", elapsed, text[:200] if text else "")
    return text


def _transcribe_whisper(audio_path: str, model_repo: str) -> str:
    """Transcribe using MLX Whisper."""
    log.info("_transcribe_whisper: Transcribing with model=%s...", model_repo)
    import mlx_whisper

    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=model_repo,
    )
    text = result.get("text", "").strip()
    log.debug("_transcribe_whisper: Raw result keys=%s", list(result.keys()))
    return text


def _transcribe_parakeet(audio_path: str, model_repo: str) -> str:
    """Transcribe using Parakeet MLX."""
    global _loaded_model, _loaded_model_repo

    if _loaded_model is None or _loaded_model_repo != model_repo:
        log.info("_transcribe_parakeet: Model not loaded or repo mismatch, loading %s...", model_repo)
        _load_parakeet_model(model_repo)

    log.info("_transcribe_parakeet: Transcribing...")
    result = _loaded_model.transcribe(audio_path)
    text = result.text.strip()
    return text
