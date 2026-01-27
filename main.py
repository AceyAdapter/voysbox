"""Voice AI Assistant — macOS menu bar app and orchestrator."""

import logging
import os
import queue
import sys
import tempfile
import threading
import time

import numpy as np
import rumps
import sounddevice as sd

import audio
import config
import hotkeys
import llm
import stt
import tts

log = logging.getLogger(__name__)


def setup_logging():
    """Configure console logging with detailed format."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(name)-12s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)

    # Quiet down noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


class VoiceAssistantApp(rumps.App):
    """Menu bar voice assistant: speak -> transcribe -> LLM -> TTS -> hear."""

    ICONS = {
        "idle": "🎙",
        "recording": "🔴",
        "transcribing": "⚡",
        "thinking": "🧠",
        "speaking": "🔊",
    }

    def __init__(self):
        log.info("=== VoiceAssistantApp initializing ===")
        super().__init__("VoiceAI", title=self.ICONS["idle"])

        # Load config
        self.cfg = config.load_config()
        log.info("Config loaded: stt=%s/%s, tts=%s/%s, llm=%s",
                 self.cfg["stt"]["engine"], self.cfg["stt"]["model"],
                 self.cfg["tts"]["engine"], self.cfg["tts"]["voice"],
                 self.cfg["llm"]["endpoint"])

        # Components
        log.info("Initializing audio components...")
        self.audio_capture = audio.AudioCapture()
        self.audio_player = audio.AudioPlayer()

        log.info("Initializing LLM client...")
        self.llm_client = llm.LLMClient(
            endpoint=self.cfg["llm"]["endpoint"],
            model=self.cfg["llm"]["model"],
            system_prompt=self.cfg["llm"]["system_prompt"],
        )

        log.info("Initializing hotkey listener...")
        self.hotkey_listener = hotkeys.HotkeyListener(
            on_start=self.on_recording_start,
            on_stop=self.on_recording_stop,
            config=self.cfg["hotkeys"],
        )

        # State
        self.state = "idle"
        self.ui_queue = queue.Queue()
        self._pipeline_lock = threading.Lock()

        # Build menu
        log.info("Building menu...")
        self._build_menu()

        # Start hotkey listener
        log.info("Starting hotkey listener...")
        self.hotkey_listener.start()

        # Preload models in background
        log.info("Preloading STT model in background...")
        stt.preload_model(
            self.cfg["stt"]["model"],
            self.cfg["stt"]["engine"],
            on_status=self._on_model_status,
        )
        log.info("Preloading TTS model in background...")
        tts.preload_tts(
            self.cfg["tts"]["engine"],
            on_status=self._on_model_status,
        )

        log.info("=== VoiceAssistantApp initialization complete ===")

    def _build_menu(self):
        """Build the rumps menu structure."""
        self.status_item = rumps.MenuItem("Status: Idle")
        self.status_item.set_callback(None)

        # --- STT Engine submenu ---
        stt_menu = rumps.MenuItem("STT Engine")
        for engine_name, models in stt.AVAILABLE_MODELS.items():
            engine_item = rumps.MenuItem(engine_name.capitalize())
            is_active_engine = self.cfg["stt"]["engine"] == engine_name
            engine_item.state = is_active_engine

            for model_label, model_repo in models.items():
                model_item = rumps.MenuItem(
                    model_label,
                    callback=self._make_stt_callback(engine_name, model_repo),
                )
                model_item.state = (
                    is_active_engine and self.cfg["stt"]["model"] == model_repo
                )
                engine_item.add(model_item)

            stt_menu.add(engine_item)

        # --- TTS Engine submenu ---
        tts_menu = rumps.MenuItem("TTS Engine")
        for engine_name, voices in tts.AVAILABLE_VOICES.items():
            engine_item = rumps.MenuItem(engine_name.capitalize())
            is_active_engine = self.cfg["tts"]["engine"] == engine_name
            engine_item.state = is_active_engine

            for voice_name in voices:
                voice_item = rumps.MenuItem(
                    voice_name,
                    callback=self._make_tts_voice_callback(engine_name, voice_name),
                )
                voice_item.state = (
                    is_active_engine and self.cfg["tts"]["voice"] == voice_name
                )
                engine_item.add(voice_item)

            tts_menu.add(engine_item)

        # --- LLM Settings submenu ---
        llm_menu = rumps.MenuItem("LLM Settings")
        llm_endpoint = rumps.MenuItem(
            f"Endpoint: {self.cfg['llm']['endpoint']}",
            callback=self._edit_endpoint,
        )
        llm_system = rumps.MenuItem(
            "Edit System Prompt...",
            callback=self._edit_system_prompt,
        )
        llm_clear = rumps.MenuItem(
            "Clear Conversation",
            callback=self._clear_conversation,
        )
        self.llm_status_item = rumps.MenuItem("Checking connection...")
        self.llm_status_item.set_callback(None)
        llm_menu.add(llm_endpoint)
        llm_menu.add(llm_system)
        llm_menu.add(llm_clear)
        llm_menu.add(self.llm_status_item)

        # --- Hotkeys submenu ---
        hold_display = hotkeys.KEY_DISPLAY.get(
            self.cfg["hotkeys"]["hold_key"],
            self.cfg["hotkeys"]["hold_key"],
        )
        toggle_display = hotkeys.KEY_DISPLAY.get(
            self.cfg["hotkeys"]["toggle_modifier"],
            self.cfg["hotkeys"]["toggle_modifier"],
        )
        hotkey_menu = rumps.MenuItem("Hotkeys")
        self.hold_key_item = rumps.MenuItem(
            f"Hold Key: {hold_display}",
            callback=self._rebind_hold_key,
        )
        self.toggle_key_item = rumps.MenuItem(
            f"Toggle Modifier: {toggle_display}",
            callback=self._rebind_toggle_key,
        )
        hotkey_menu.add(self.hold_key_item)
        hotkey_menu.add(self.toggle_key_item)

        # --- Audio device submenus ---
        input_menu = rumps.MenuItem("Input Device")
        output_menu = rumps.MenuItem("Output Device")
        self._populate_device_menus(input_menu, output_menu)

        # Assemble menu
        self.menu = [
            self.status_item,
            None,  # separator
            stt_menu,
            tts_menu,
            llm_menu,
            None,
            hotkey_menu,
            None,
            input_menu,
            output_menu,
            None,
        ]

        log.info("Menu built successfully")

        # Check LLM connection in background
        threading.Thread(target=self._check_llm_connection, daemon=True).start()

    def _populate_device_menus(self, input_menu, output_menu):
        """Populate input/output device menus from sounddevice."""
        try:
            devices = sd.query_devices()
            log.info("Found %d audio devices", len(devices))
            input_count = 0
            output_count = 0
            for i, dev in enumerate(devices):
                if dev["max_input_channels"] > 0:
                    item = rumps.MenuItem(
                        dev["name"],
                        callback=self._make_input_device_callback(i),
                    )
                    input_menu.add(item)
                    input_count += 1
                    log.debug("  Input device [%d]: %s (%d ch)",
                              i, dev["name"], dev["max_input_channels"])
                if dev["max_output_channels"] > 0:
                    item = rumps.MenuItem(
                        dev["name"],
                        callback=self._make_output_device_callback(i),
                    )
                    output_menu.add(item)
                    output_count += 1
                    log.debug("  Output device [%d]: %s (%d ch)",
                              i, dev["name"], dev["max_output_channels"])
            log.info("Audio devices: %d inputs, %d outputs", input_count, output_count)
        except Exception as e:
            log.error("Failed to query audio devices: %s", e, exc_info=True)
            input_menu.add(rumps.MenuItem("(no devices found)"))
            output_menu.add(rumps.MenuItem("(no devices found)"))

    # --- Pipeline ---

    def on_recording_start(self):
        """Called by hotkey listener when recording should begin."""
        log.info(">>> on_recording_start() — current state=%s", self.state)

        # Interrupt TTS playback if speaking
        if self.state == "speaking":
            log.info("Interrupting TTS playback to start recording")
            self.audio_player.stop()

        self._set_state("recording")
        self.audio_capture.start()

    def on_recording_stop(self):
        """Called by hotkey listener when recording should end."""
        log.info(">>> on_recording_stop() — current state=%s", self.state)
        audio_data = self.audio_capture.stop()

        if len(audio_data) == 0:
            log.warning("No audio data captured, returning to idle")
            self._set_state("idle")
            return

        duration = len(audio_data) / audio.SAMPLE_RATE
        log.info("Captured %.2fs of audio (%d samples), launching pipeline",
                 duration, len(audio_data))
        threading.Thread(
            target=self._pipeline, args=(audio_data,), daemon=True
        ).start()

    def _pipeline(self, audio_data: np.ndarray):
        """Run the full STT -> LLM -> TTS -> Play pipeline."""
        log.info("=== Pipeline starting ===")
        pipeline_t0 = time.time()

        if not self._pipeline_lock.acquire(blocking=False):
            log.warning("Pipeline: Another pipeline is already running, skipping")
            return

        try:
            # 1. Transcribe
            log.info("--- Pipeline step 1/4: Transcribe ---")
            self._set_state("transcribing")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = tmp.name
            self.audio_capture.save_wav(audio_data, wav_path)
            log.info("Pipeline: Saved audio to temp file: %s", wav_path)

            try:
                t0 = time.time()
                text = stt.transcribe(
                    wav_path,
                    self.cfg["stt"]["engine"],
                    self.cfg["stt"]["model"],
                )
                log.info("Pipeline: STT completed in %.2fs — text=%r",
                         time.time() - t0, text[:200] if text else "")
            finally:
                try:
                    os.unlink(wav_path)
                    log.debug("Pipeline: Temp WAV file cleaned up")
                except OSError:
                    pass

            if not text.strip():
                log.warning("Pipeline: Empty transcription, returning to idle")
                self._set_state("idle")
                return

            # 2. LLM
            log.info("--- Pipeline step 2/4: LLM ---")
            self._set_state("thinking")
            t0 = time.time()
            try:
                response = self.llm_client.chat(text)
                log.info("Pipeline: LLM responded in %.2fs — response=%r",
                         time.time() - t0, response[:200])
            except Exception as e:
                log.error("Pipeline: LLM request failed: %s", e, exc_info=True)
                response = "Sorry, the language model is not responding."

            # 3. TTS
            log.info("--- Pipeline step 3/4: TTS ---")
            self._set_state("speaking")
            try:
                t0 = time.time()
                audio_out, sr = tts.synthesize(
                    response,
                    self.cfg["tts"]["engine"],
                    self.cfg["tts"]["voice"],
                )
                log.info("Pipeline: TTS completed in %.2fs — %d samples at %d Hz (%.2fs audio)",
                         time.time() - t0, len(audio_out), sr,
                         len(audio_out) / sr if len(audio_out) > 0 else 0)

                # 4. Play
                log.info("--- Pipeline step 4/4: Play ---")
                t0 = time.time()
                self.audio_player.play(audio_out, sr)
                log.info("Pipeline: Playback finished in %.2fs", time.time() - t0)
            except Exception as e:
                log.error("Pipeline: TTS/playback failed: %s", e, exc_info=True)

            self._set_state("idle")
            total_elapsed = time.time() - pipeline_t0
            log.info("=== Pipeline complete — total elapsed: %.2fs ===", total_elapsed)
        finally:
            self._pipeline_lock.release()

    # --- State management ---

    def _set_state(self, state: str):
        """Update app state and queue a UI update."""
        old_state = self.state
        self.state = state
        icon = self.ICONS.get(state, self.ICONS["idle"])
        status_text = state.capitalize()
        self.ui_queue.put((icon, status_text))
        log.info("State: %s -> %s", old_state, state)

    @rumps.timer(0.1)
    def _process_ui_queue(self, _):
        """Poll the UI queue and update menu bar."""
        while not self.ui_queue.empty():
            try:
                icon, status_text = self.ui_queue.get_nowait()
                self.title = icon
                self.status_item.title = f"Status: {status_text}"
            except queue.Empty:
                break

    # --- Model status ---

    def _on_model_status(self, status: str):
        """Called by STT/TTS preload with status updates."""
        log.info("Model status update: %s", status)
        self.ui_queue.put((self.ICONS.get(self.state, "🎙"), status))

    # --- LLM connection check ---

    def _check_llm_connection(self):
        """Check LLM availability and update menu."""
        log.info("Checking LLM connection...")
        available = self.llm_client.is_available()
        status = "Connected ✓" if available else "Not connected ✗"
        log.info("LLM connection status: %s", status)
        self.llm_status_item.title = f"Connection: {status}"

    # --- Menu callbacks ---

    def _make_stt_callback(self, engine: str, model_repo: str):
        """Create a callback for selecting an STT engine/model."""
        def callback(sender):
            log.info("Menu: STT engine changed to %s / %s", engine, model_repo)
            self.cfg["stt"]["engine"] = engine
            self.cfg["stt"]["model"] = model_repo
            config.save_config(self.cfg)
            stt.preload_model(
                model_repo,
                engine,
                on_status=self._on_model_status,
            )
            rumps.notification(
                "Voice AI",
                "STT Engine Changed",
                f"{engine.capitalize()}: {model_repo.split('/')[-1]}",
            )
        return callback

    def _make_tts_voice_callback(self, engine: str, voice: str):
        """Create a callback for selecting a TTS engine/voice."""
        def callback(sender):
            log.info("Menu: TTS voice changed to %s / %s", engine, voice)
            self.cfg["tts"]["engine"] = engine
            self.cfg["tts"]["voice"] = voice
            config.save_config(self.cfg)
            if engine != tts._loaded_tts_engine:
                log.info("Menu: TTS engine changed, preloading new engine...")
                tts.preload_tts(
                    engine,
                    on_status=self._on_model_status,
                )
            rumps.notification(
                "Voice AI",
                "TTS Voice Changed",
                f"{engine.capitalize()}: {voice}",
            )
        return callback

    def _make_input_device_callback(self, device_index: int):
        """Create a callback for selecting an input audio device."""
        def callback(sender):
            log.info("Menu: Input device changed to index %d", device_index)
            self.audio_capture.device = device_index
        return callback

    def _make_output_device_callback(self, device_index: int):
        """Create a callback for selecting an output audio device."""
        def callback(sender):
            log.info("Menu: Output device changed to index %d", device_index)
            self.audio_player.device = device_index
        return callback

    def _edit_endpoint(self, sender):
        """Open a dialog to edit the LLM endpoint."""
        response = rumps.Window(
            message="Enter LLM API endpoint:",
            title="LLM Endpoint",
            default_text=self.cfg["llm"]["endpoint"],
            ok="Save",
            cancel="Cancel",
        ).run()
        if response.clicked:
            new_endpoint = response.text.strip()
            log.info("Menu: LLM endpoint changed to %s", new_endpoint)
            self.cfg["llm"]["endpoint"] = new_endpoint
            config.save_config(self.cfg)
            self.llm_client.endpoint = new_endpoint
            sender.title = f"Endpoint: {new_endpoint}"
            threading.Thread(target=self._check_llm_connection, daemon=True).start()

    def _edit_system_prompt(self, sender):
        """Open a dialog to edit the system prompt."""
        response = rumps.Window(
            message="Enter system prompt:",
            title="System Prompt",
            default_text=self.cfg["llm"]["system_prompt"],
            ok="Save",
            cancel="Cancel",
        ).run()
        if response.clicked:
            new_prompt = response.text.strip()
            log.info("Menu: System prompt changed (%d chars)", len(new_prompt))
            self.cfg["llm"]["system_prompt"] = new_prompt
            config.save_config(self.cfg)
            self.llm_client.system_prompt = new_prompt

    def _clear_conversation(self, sender):
        """Clear LLM conversation history."""
        log.info("Menu: Clearing conversation history")
        self.llm_client.clear_history()
        rumps.notification("Voice AI", "Conversation Cleared", "History has been reset.")

    def _rebind_hold_key(self, sender):
        """Enter hotkey capture mode for the hold key."""
        log.info("Menu: Rebinding hold key — entering capture mode")
        self.hold_key_item.title = "Hold Key: Press a key..."
        def _on_capture(key_name):
            log.info("Menu: Hold key rebound to %s", key_name)
            self.cfg["hotkeys"]["hold_key"] = key_name
            config.save_config(self.cfg)
            self.hotkey_listener.update_keys(self.cfg["hotkeys"])
            display = hotkeys.KEY_DISPLAY.get(key_name, key_name)
            self.hold_key_item.title = f"Hold Key: {display}"
        self.hotkey_listener.start_capture(_on_capture)

    def _rebind_toggle_key(self, sender):
        """Enter hotkey capture mode for the toggle modifier."""
        log.info("Menu: Rebinding toggle modifier — entering capture mode")
        self.toggle_key_item.title = "Toggle Modifier: Press a key..."
        def _on_capture(key_name):
            log.info("Menu: Toggle modifier rebound to %s", key_name)
            self.cfg["hotkeys"]["toggle_modifier"] = key_name
            config.save_config(self.cfg)
            self.hotkey_listener.update_keys(self.cfg["hotkeys"])
            display = hotkeys.KEY_DISPLAY.get(key_name, key_name)
            self.toggle_key_item.title = f"Toggle Modifier: {display}"
        self.hotkey_listener.start_capture(_on_capture)

    def terminate(self):
        """Clean up on exit."""
        log.info("VoiceAssistantApp.terminate() — cleaning up")
        self.hotkey_listener.stop()
        self.audio_player.stop()
        config.release_single_instance_lock()


def main():
    setup_logging()
    log.info("=" * 60)
    log.info("Voice AI Assistant starting")
    log.info("=" * 60)

    if not config.acquire_single_instance_lock():
        log.error("Another instance is already running — exiting")
        rumps.alert(
            title="Voice AI",
            message="Another instance is already running.",
        )
        sys.exit(1)

    try:
        app = VoiceAssistantApp()
        log.info("Starting rumps event loop...")
        app.run()
    except Exception as e:
        log.critical("Fatal error: %s", e, exc_info=True)
        raise
    finally:
        log.info("Shutting down...")
        config.release_single_instance_lock()


if __name__ == "__main__":
    main()
