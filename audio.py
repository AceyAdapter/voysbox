"""Audio capture and playback for Voice AI Assistant."""

import logging
import threading
import wave

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
MAX_DURATION_S = 300  # 5 min safety limit


class AudioCapture:
    """Callback-based audio recording via sounddevice InputStream."""

    def __init__(self, device=None):
        self.device = device
        self.stream = None
        self.audio_chunks: list[np.ndarray] = []
        self.recording = False
        log.info("AudioCapture initialized (device=%s)", device)

    def start(self):
        """Start recording audio."""
        log.info("AudioCapture.start() — device=%s, rate=%d, channels=%d",
                 self.device, SAMPLE_RATE, CHANNELS)
        self.audio_chunks = []
        self.recording = True
        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                device=self.device,
                callback=self._audio_callback,
            )
            self.stream.start()
            log.info("AudioCapture: InputStream started successfully")
        except Exception as e:
            log.error("AudioCapture: Failed to start InputStream: %s", e, exc_info=True)
            self.recording = False
            raise

    def stop(self) -> np.ndarray:
        """Stop recording and return concatenated float32 audio array."""
        log.info("AudioCapture.stop() — chunks collected: %d", len(self.audio_chunks))
        self.recording = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
                log.debug("AudioCapture: InputStream stopped and closed")
            except Exception as e:
                log.warning("AudioCapture: Error stopping stream: %s", e)
            self.stream = None

        if not self.audio_chunks:
            log.warning("AudioCapture: No audio chunks recorded — returning empty array")
            return np.array([], dtype=np.float32)

        audio = np.concatenate(self.audio_chunks)
        duration_s = len(audio) / SAMPLE_RATE
        log.info("AudioCapture: Recorded %.2fs of audio (%d samples)", duration_s, len(audio))

        # Enforce max duration
        max_samples = MAX_DURATION_S * SAMPLE_RATE
        if len(audio) > max_samples:
            log.warning("AudioCapture: Truncating audio from %d to %d samples (max %ds)",
                        len(audio), max_samples, MAX_DURATION_S)
            audio = audio[:max_samples]

        self.audio_chunks = []
        return audio

    def _audio_callback(self, indata, frames, time_info, status):
        """Sounddevice input stream callback."""
        if status:
            log.warning("AudioCapture callback status: %s", status)
        if self.recording:
            self.audio_chunks.append(indata[:, 0].copy())

    def save_wav(self, audio: np.ndarray, path: str) -> str:
        """Save float32 audio array to WAV file (int16). Returns the path."""
        log.info("AudioCapture.save_wav() — saving %d samples to %s", len(audio), path)
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        with wave.open(path, "w") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())
        log.debug("AudioCapture: WAV saved (%d bytes)", len(audio_int16.tobytes()))
        return path


class AudioPlayer:
    """Play TTS audio output via sounddevice."""

    def __init__(self, device=None):
        self.device = device
        self.playing = False
        self._stop_flag = threading.Event()
        self._play_thread = None
        log.info("AudioPlayer initialized (device=%s)", device)

    def play(self, audio: np.ndarray, sample_rate: int):
        """Blocking playback of audio array."""
        log.info("AudioPlayer.play() — %d samples at %d Hz (%.2fs), device=%s",
                 len(audio), sample_rate, len(audio) / sample_rate, self.device)
        self._stop_flag.clear()
        self.playing = True

        # Resample if needed to match output device
        output_sr = self._get_output_sample_rate()
        if sample_rate != output_sr:
            log.info("AudioPlayer: Resampling from %d Hz to %d Hz", sample_rate, output_sr)
            audio = self._resample(audio, sample_rate, output_sr)
            sample_rate = output_sr
            log.debug("AudioPlayer: Resampled to %d samples", len(audio))

        try:
            sd.play(audio, samplerate=sample_rate, device=self.device)
            log.debug("AudioPlayer: sd.play() called, polling for completion...")
            while sd.get_stream().active and not self._stop_flag.is_set():
                sd.sleep(50)
            if self._stop_flag.is_set():
                log.info("AudioPlayer: Playback interrupted by stop flag")
                sd.stop()
            else:
                log.info("AudioPlayer: Playback completed normally")
        except Exception as e:
            log.error("AudioPlayer: Playback error: %s", e, exc_info=True)
        finally:
            self.playing = False

    def play_async(self, audio: np.ndarray, sample_rate: int):
        """Non-blocking playback in a background thread."""
        log.info("AudioPlayer.play_async() — launching background playback")
        self._play_thread = threading.Thread(
            target=self.play, args=(audio, sample_rate), daemon=True
        )
        self._play_thread.start()

    def stop(self):
        """Interrupt playback immediately."""
        log.info("AudioPlayer.stop() — interrupting playback (was_playing=%s)", self.playing)
        self._stop_flag.set()
        try:
            sd.stop()
        except Exception as e:
            log.warning("AudioPlayer: Error during sd.stop(): %s", e)
        self.playing = False

    def _get_output_sample_rate(self) -> int:
        """Get the default output device sample rate."""
        try:
            if self.device is not None:
                info = sd.query_devices(self.device)
            else:
                info = sd.query_devices(kind="output")
            sr = int(info["default_samplerate"])
            log.debug("AudioPlayer: Output device sample rate = %d Hz", sr)
            return sr
        except Exception as e:
            log.warning("AudioPlayer: Could not query output device, defaulting to 44100 Hz: %s", e)
            return 44100

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        from scipy.signal import resample

        num_samples = int(len(audio) * target_sr / orig_sr)
        return resample(audio, num_samples).astype(np.float32)
