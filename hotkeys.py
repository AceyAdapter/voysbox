"""Global hotkey listener for Voice AI Assistant."""

import logging

from pynput import keyboard

log = logging.getLogger(__name__)

# Map config key names to pynput Key objects
KEY_MAP = {
    "alt_l": keyboard.Key.alt_l,
    "alt_r": keyboard.Key.alt_r,
    "alt": keyboard.Key.alt,
    "ctrl_l": keyboard.Key.ctrl_l,
    "ctrl_r": keyboard.Key.ctrl_r,
    "ctrl": keyboard.Key.ctrl,
    "shift_l": keyboard.Key.shift_l,
    "shift_r": keyboard.Key.shift_r,
    "shift": keyboard.Key.shift,
    "cmd_l": keyboard.Key.cmd_l,
    "cmd_r": keyboard.Key.cmd_r,
    "cmd": keyboard.Key.cmd,
    "space": keyboard.Key.space,
    "caps_lock": keyboard.Key.caps_lock,
    "tab": keyboard.Key.tab,
    "f1": keyboard.Key.f1,
    "f2": keyboard.Key.f2,
    "f3": keyboard.Key.f3,
    "f4": keyboard.Key.f4,
    "f5": keyboard.Key.f5,
    "f6": keyboard.Key.f6,
    "f7": keyboard.Key.f7,
    "f8": keyboard.Key.f8,
    "f9": keyboard.Key.f9,
    "f10": keyboard.Key.f10,
    "f11": keyboard.Key.f11,
    "f12": keyboard.Key.f12,
}

# Human-readable display names
KEY_DISPLAY = {
    "alt_l": "Left Option",
    "alt_r": "Right Option",
    "alt": "Option",
    "ctrl_l": "Left Control",
    "ctrl_r": "Right Control",
    "ctrl": "Control",
    "shift_l": "Left Shift",
    "shift_r": "Right Shift",
    "shift": "Shift",
    "cmd_l": "Left Command",
    "cmd_r": "Right Command",
    "cmd": "Command",
    "space": "Space",
    "caps_lock": "Caps Lock",
    "tab": "Tab",
    "f1": "F1",
    "f2": "F2",
    "f3": "F3",
    "f4": "F4",
    "f5": "F5",
    "f6": "F6",
    "f7": "F7",
    "f8": "F8",
    "f9": "F9",
    "f10": "F10",
    "f11": "F11",
    "f12": "F12",
}


class HotkeyListener:
    """Listens for global hotkeys to control recording.

    Supports two modes:
    - Hold mode: Hold the hold_key to record, release to stop.
    - Toggle mode: Press toggle_modifier + hold_key to start, press again to stop.
    """

    def __init__(self, on_start, on_stop, config: dict):
        self.on_start = on_start
        self.on_stop = on_stop
        self.hold_key = config.get("hold_key", "alt_r")
        self.toggle_modifier = config.get("toggle_modifier", "alt_l")
        self.listener = None
        self.modifier_pressed = False
        self.recording = False
        self._toggle_mode = False
        self._capture_callback = None
        log.info("HotkeyListener initialized — hold_key=%s (%s), toggle_modifier=%s (%s)",
                 self.hold_key, KEY_DISPLAY.get(self.hold_key, self.hold_key),
                 self.toggle_modifier, KEY_DISPLAY.get(self.toggle_modifier, self.toggle_modifier))

    def start(self):
        """Start listening for hotkeys."""
        log.info("HotkeyListener.start() — starting keyboard listener")
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self.listener.daemon = True
        self.listener.start()
        log.info("HotkeyListener: Keyboard listener started")

    def stop(self):
        """Stop the hotkey listener."""
        log.info("HotkeyListener.stop()")
        if self.listener:
            self.listener.stop()
            self.listener = None
            log.info("HotkeyListener: Keyboard listener stopped")

    def update_keys(self, config: dict):
        """Update hotkey bindings from config."""
        self.hold_key = config.get("hold_key", "alt_r")
        self.toggle_modifier = config.get("toggle_modifier", "alt_l")
        log.info("HotkeyListener: Keys updated — hold_key=%s, toggle_modifier=%s",
                 self.hold_key, self.toggle_modifier)

    def _on_press(self, key):
        """Handle key press events."""
        log.debug("HotkeyListener: Key pressed: %s", key)

        # Capture mode for rebinding
        if self._capture_callback:
            log.info("HotkeyListener: In capture mode — capturing key: %s", key)
            self._capture_key(key)
            return

        # Track modifier state
        if self._key_matches(key, self.toggle_modifier):
            self.modifier_pressed = True
            log.debug("HotkeyListener: Toggle modifier pressed")

        # Hold key pressed
        if self._key_matches(key, self.hold_key):
            if self.modifier_pressed:
                # Toggle mode
                if self.recording:
                    log.info("HotkeyListener: TOGGLE OFF — stopping recording")
                    self.recording = False
                    self._toggle_mode = False
                    self.on_stop()
                else:
                    log.info("HotkeyListener: TOGGLE ON — starting recording")
                    self.recording = True
                    self._toggle_mode = True
                    self.on_start()
            elif not self.recording:
                # Hold mode — start recording
                log.info("HotkeyListener: HOLD START — starting recording")
                self.recording = True
                self._toggle_mode = False
                self.on_start()

    def _on_release(self, key):
        """Handle key release events."""
        log.debug("HotkeyListener: Key released: %s", key)

        if self._capture_callback:
            return

        # Track modifier state
        if self._key_matches(key, self.toggle_modifier):
            self.modifier_pressed = False
            log.debug("HotkeyListener: Toggle modifier released")

        # Hold key released — stop recording if in hold mode
        if self._key_matches(key, self.hold_key):
            if self.recording and not self._toggle_mode:
                log.info("HotkeyListener: HOLD RELEASE — stopping recording")
                self.recording = False
                self.on_stop()

    def _key_matches(self, pressed_key, key_name: str) -> bool:
        """Check if a pressed key matches a key name from config."""
        target = KEY_MAP.get(key_name)
        if target is None:
            try:
                return hasattr(pressed_key, "char") and pressed_key.char == key_name
            except AttributeError:
                return False
        return pressed_key == target

    def start_capture(self, callback):
        """Enter hotkey capture mode for rebinding. callback(key_name) is called with the captured key."""
        log.info("HotkeyListener: Entering capture mode")
        self._capture_callback = callback

    def stop_capture(self):
        """Exit hotkey capture mode."""
        log.info("HotkeyListener: Exiting capture mode")
        self._capture_callback = None

    def _capture_key(self, key):
        """Process a captured key during rebinding."""
        # Reverse lookup: pynput key -> config key name
        for name, pynput_key in KEY_MAP.items():
            if key == pynput_key:
                log.info("HotkeyListener: Captured special key: %s (%s)",
                         name, KEY_DISPLAY.get(name, name))
                cb = self._capture_callback
                self._capture_callback = None
                cb(name)
                return

        # Character key
        if hasattr(key, "char") and key.char:
            log.info("HotkeyListener: Captured character key: %s", key.char)
            cb = self._capture_callback
            self._capture_callback = None
            cb(key.char)
        else:
            log.warning("HotkeyListener: Unrecognized key in capture mode: %s", key)
