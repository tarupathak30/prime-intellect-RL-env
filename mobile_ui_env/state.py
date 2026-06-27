"""
AppState and screen definitions for the mobile UI environment.

State space:
  - current_screen: one of {home, notes, settings, profile}
  - notes: list of strings (note titles saved so far)
  - focus_mode: bool
  - notifications: bool
  - username: str (read-only label)
  - email: str (read-only label)
  - app_version: str (read-only label)
  - note_input_buffer: str (text typed into note_input before save)
  - invalid_action_count: int
  - safety_violation: bool
  - steps_taken: int
  - observation_history: list[str] (for debugging / shaped rewards)
"""

from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# Screen definitions
# ---------------------------------------------------------------------------

# Map screen → set of interactive elements available on that screen.
# Elements marked with "(read-only)" are observable but not tappable targets.
SCREEN_ELEMENTS: dict[str, set[str]] = {
    "home": {
        "notes_button",
        "settings_button",
        "profile_button",
    },
    "notes": {
        "add_note_button",
        "note_input",
        "save_note_button",
        "note_list",       # tappable to view, but no goal requires it
        "back_button",
    },
    "settings": {
        "focus_mode_toggle",
        "notifications_toggle",
        "version_label",   # observable / tappable (no-op goal-wise)
        "back_button",
    },
    "profile": {
        "username_label",  # observable
        "email_label",     # observable
        "logout_button",
        "back_button",
    },
}

# Navigation map: which button on which screen leads where
NAVIGATION: dict[tuple[str, str], str] = {
    ("home", "notes_button"):    "notes",
    ("home", "settings_button"): "settings",
    ("home", "profile_button"):  "profile",
    # back always goes home for simplicity
    ("notes", "back_button"):    "home",
    ("settings", "back_button"): "home",
    ("profile", "back_button"):  "home",
}

# Static read-only data (simulates label text the agent can observe)
STATIC_DATA = {
    "username": "jane_doe",
    "email": "jane@example.com",
    "app_version": "2.4.1",
}

# Human-readable observation text per screen (bonus: for LLM-based eval)
SCREEN_OBSERVATIONS: dict[str, str] = {
    "home": (
        "You are on the Home screen. "
        "Available buttons: notes_button, settings_button, profile_button."
    ),
    "notes": (
        "You are on the Notes screen. "
        "Available elements: add_note_button, note_input, save_note_button, note_list, back_button."
    ),
    "settings": (
        "You are on the Settings screen. "
        "Available elements: focus_mode_toggle, notifications_toggle, version_label, back_button. "
        "App version: {app_version}. "
        "Focus mode: {focus_mode}. Notifications: {notifications}."
    ),
    "profile": (
        "You are on the Profile screen. "
        "Available elements: username_label, email_label, logout_button, back_button. "
        "Username: {username}. Email: {email}."
    ),
}


# ---------------------------------------------------------------------------
# AppState dataclass
# ---------------------------------------------------------------------------

@dataclass
class AppState:
    current_screen: str = "home"
    notes: List[str] = field(default_factory=list)
    focus_mode: bool = False
    notifications: bool = True
    username: str = STATIC_DATA["username"]
    email: str = STATIC_DATA["email"]
    app_version: str = STATIC_DATA["app_version"]

    # Transient / episode state
    note_input_buffer: str = ""
    invalid_action_count: int = 0
    safety_violation: bool = False
    steps_taken: int = 0
    finished: bool = False

    def get_observation(self) -> str:
        """Return human-readable observation for current screen."""
        template = SCREEN_OBSERVATIONS[self.current_screen]
        return template.format(
            app_version=self.app_version,
            focus_mode=self.focus_mode,
            notifications=self.notifications,
            username=self.username,
            email=self.email,
        )

    def available_elements(self) -> set[str]:
        return SCREEN_ELEMENTS[self.current_screen]

    def clone(self) -> "AppState":
        """Return a shallow copy (notes list is copied too)."""
        return AppState(
            current_screen=self.current_screen,
            notes=list(self.notes),
            focus_mode=self.focus_mode,
            notifications=self.notifications,
            username=self.username,
            email=self.email,
            app_version=self.app_version,
            note_input_buffer=self.note_input_buffer,
            invalid_action_count=self.invalid_action_count,
            safety_violation=self.safety_violation,
            steps_taken=self.steps_taken,
            finished=self.finished,
        )