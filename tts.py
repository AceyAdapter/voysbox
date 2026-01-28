"""Text-to-speech engines for Voice AI Assistant."""

import logging
import threading
import time

import numpy as np

log = logging.getLogger(__name__)

_loaded_tts_model = None
_loaded_tts_engine = None
_tts_operation_in_progress = False
_tts_lock = threading.Lock()

AVAILABLE_VOICES = {
    "kokoro": [
        "af_heart",
        "af_bella",
        "af_nicole",
        "af_sarah",
        "af_sky",
        "am_adam",
        "am_michael",
        "bf_emma",
        "bf_isabella",
        "bm_george",
        "bm_lewis",
    ],
    "supertonic": [],  # Populated at runtime from library
}


def preload_tts(engine: str, on_status=None, on_complete=None):
    """Preload a TTS model in a background thread."""
    global _tts_operation_in_progress

    log.info("preload_tts() called — engine=%s", engine)

    if _tts_operation_in_progress:
        log.warning("preload_tts: Another TTS operation is already in progress, skipping")
        if on_complete:
            on_complete(False)
        return

    def _load():
        global _loaded_tts_model, _loaded_tts_engine, _tts_operation_in_progress
        _tts_operation_in_progress = True
        t0 = time.time()
        try:
            if on_status:
                on_status(f"Loading {engine} TTS model...")
            log.info("preload_tts: Loading %s TTS model...", engine)

            if engine == "kokoro":
                _load_kokoro()
            elif engine == "supertonic":
                _load_supertonic()
            else:
                raise ValueError(f"Unknown TTS engine: {engine}")

            _loaded_tts_engine = engine

            elapsed = time.time() - t0
            log.info("preload_tts: %s TTS model loaded in %.2fs", engine, elapsed)
            if on_status:
                on_status("TTS model loaded")
            if on_complete:
                on_complete(True)
        except Exception as e:
            elapsed = time.time() - t0
            log.error("preload_tts: Failed to load %s TTS after %.2fs: %s",
                      engine, elapsed, e, exc_info=True)
            if on_status:
                on_status(f"Failed to load TTS: {e}")
            if on_complete:
                on_complete(False)
        finally:
            _tts_operation_in_progress = False

    threading.Thread(target=_load, daemon=True).start()


def _load_kokoro():
    """Load and cache the Kokoro TTS pipeline."""
    global _loaded_tts_model

    # Try multiple loading strategies
    # Strategy 1: KokoroPipeline from mlx_audio (direct import with properly loaded model)
    try:
        log.info("_load_kokoro: Trying KokoroPipeline from mlx_audio.tts.models.kokoro...")
        from mlx_audio.tts.models.kokoro import KokoroPipeline
        from mlx_audio.tts.utils import load_model as load_tts_model
        log.info("_load_kokoro: Loading model weights from mlx-community/Kokoro-82M-bf16...")
        model = load_tts_model("mlx-community/Kokoro-82M-bf16")
        log.info("_load_kokoro: Model loaded, creating pipeline (lang_code='a')...")
        pipeline = KokoroPipeline(
            lang_code="a",
            model=model,
            repo_id="mlx-community/Kokoro-82M-bf16",
        )
        _loaded_tts_model = ("kokoro_pipeline", pipeline)
        log.info("_load_kokoro: KokoroPipeline loaded successfully")
        return
    except Exception as e:
        log.warning("_load_kokoro: KokoroPipeline failed: %s", e, exc_info=True)

    # Strategy 2: load_model from mlx_audio (generic API)
    try:
        log.info("_load_kokoro: Trying load_model from mlx_audio.tts.utils...")
        from mlx_audio.tts.utils import load_model
        model = load_model("mlx-community/Kokoro-82M-bf16")
        _loaded_tts_model = ("mlx_audio_model", model)
        log.info("_load_kokoro: load_model succeeded")
        return
    except Exception as e:
        log.warning("_load_kokoro: load_model failed: %s", e, exc_info=True)

    # Strategy 3: Standalone kokoro package (KPipeline)
    try:
        log.info("_load_kokoro: Trying standalone kokoro package (KPipeline)...")
        from kokoro import KPipeline
        pipeline = KPipeline(lang_code="a")
        _loaded_tts_model = ("kokoro_standalone", pipeline)
        log.info("_load_kokoro: Standalone KPipeline loaded successfully")
        return
    except Exception as e:
        log.warning("_load_kokoro: Standalone kokoro package failed: %s", e, exc_info=True)

    raise RuntimeError(
        "Failed to load Kokoro TTS model. Tried: "
        "mlx_audio KokoroPipeline, mlx_audio load_model, standalone kokoro package. "
        "Install one of: pip install mlx-audio   OR   pip install kokoro"
    )


def _load_supertonic():
    """Load and cache the Supertonic TTS model."""
    global _loaded_tts_model
    log.info("_load_supertonic: Importing supertonic...")
    import supertonic  # noqa: F401
    _loaded_tts_model = ("supertonic", True)
    log.info("_load_supertonic: Supertonic loaded successfully")


def synthesize(text: str, engine: str, voice: str = "af_heart") -> tuple[np.ndarray, int]:
    """Synthesize text to audio. Returns (audio_array, sample_rate)."""
    log.info("synthesize() called — engine=%s, voice=%s, text=%r (%d chars)",
             engine, voice, text[:80], len(text))
    t0 = time.time()

    with _tts_lock:
        if engine == "kokoro":
            result = _synthesize_kokoro(text, voice)
        elif engine == "supertonic":
            result = _synthesize_supertonic(text, voice)
        else:
            raise ValueError(f"Unknown TTS engine: {engine}")

    audio, sr = result
    elapsed = time.time() - t0
    duration = len(audio) / sr if len(audio) > 0 else 0
    log.info("synthesize: Completed in %.2fs — output: %d samples, %d Hz, %.2fs audio",
             elapsed, len(audio), sr, duration)
    return result


def _synthesize_kokoro(text: str, voice: str) -> tuple[np.ndarray, int]:
    """Synthesize using Kokoro (with fallback strategies)."""
    global _loaded_tts_model

    if _loaded_tts_model is None or _loaded_tts_engine != "kokoro":
        log.info("_synthesize_kokoro: Model not loaded, loading now...")
        _load_kokoro()

    model_type, model = _loaded_tts_model
    log.info("_synthesize_kokoro: Using %s backend", model_type)

    if model_type == "kokoro_pipeline":
        # mlx_audio KokoroPipeline: yields (graphemes, phonemes, audio)
        # audio is an mx.array with shape (1, time) — squeeze to 1D
        log.debug("_synthesize_kokoro: Generating via KokoroPipeline...")
        audio_segments = []
        for graphemes, phonemes, audio_chunk in model(text, voice=voice, speed=1.0):
            chunk = np.array(audio_chunk).squeeze()
            log.debug("_synthesize_kokoro: Got chunk — %d samples", len(chunk))
            audio_segments.append(chunk)

        if not audio_segments:
            log.warning("_synthesize_kokoro: No audio segments generated")
            return np.array([], dtype=np.float32), 24000

        audio = np.concatenate(audio_segments)
        log.info("_synthesize_kokoro: Concatenated %d segments -> %d samples", len(audio_segments), len(audio))
        return audio, 24000

    elif model_type == "mlx_audio_model":
        # mlx_audio load_model: yields Result objects with .audio
        log.debug("_synthesize_kokoro: Generating via mlx_audio model.generate()...")
        audio_segments = []
        for result in model.generate(text, voice=voice, speed=1.0, lang_code="a"):
            chunk = result.audio if hasattr(result, "audio") else result
            if hasattr(chunk, "numpy"):
                chunk = np.array(chunk)
            log.debug("_synthesize_kokoro: Got chunk — %d samples", len(chunk))
            audio_segments.append(chunk)

        if not audio_segments:
            log.warning("_synthesize_kokoro: No audio segments generated")
            return np.array([], dtype=np.float32), 24000

        audio = np.concatenate(audio_segments)
        log.info("_synthesize_kokoro: Concatenated %d segments -> %d samples", len(audio_segments), len(audio))
        return audio, 24000

    elif model_type == "kokoro_standalone":
        # Standalone kokoro package: yields (graphemes, phonemes, audio)
        log.debug("_synthesize_kokoro: Generating via standalone KPipeline...")
        audio_segments = []
        for graphemes, phonemes, audio_chunk in model(text, voice=voice, speed=1.0):
            log.debug("_synthesize_kokoro: Got chunk — %d samples", len(audio_chunk))
            audio_segments.append(audio_chunk)

        if not audio_segments:
            log.warning("_synthesize_kokoro: No audio segments generated")
            return np.array([], dtype=np.float32), 24000

        audio = np.concatenate(audio_segments)
        log.info("_synthesize_kokoro: Concatenated %d segments -> %d samples", len(audio_segments), len(audio))
        return audio, 24000

    else:
        raise RuntimeError(f"Unknown Kokoro model type: {model_type}")


def _synthesize_supertonic(text: str, voice: str) -> tuple[np.ndarray, int]:
    """Synthesize using Supertonic."""
    log.info("_synthesize_supertonic: Synthesizing with voice=%s...", voice)
    from supertonic import synthesize as st_synthesize

    audio = st_synthesize(text, voice=voice)
    log.info("_synthesize_supertonic: Got %d samples", len(audio))
    return audio, 24000
