from dataclasses import dataclass, field
from typing import List


# Screen definitions

# Elements marked with "(read-only)" are observable but not tappable targets.
SCREEN_ELEMENTS: dict[str, set[str]] = {
    # this maps screen ---> buttons/elements available 
    # basically, on this screen, what can agent interact with? 
    # when the agent is here, current_screen = "home", 
    # available actions are tap notes_button, tap settings_button, tap_profile_button
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
    # the above means, if current_screen == "home" and agent taps "notes_button", then the next state is current_scree = "notes"
    # (state, action) -> next_state(P(s'|s, a))

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
    # this is the text representation of the screen. 
    # why? because maybe later the agent is an LLM. 

    # instead of current_screen="home", 
    # we give, 
    # You are on the Home screen. 
    # Available buttons: 
    # notes_button, settings_button, profile_button.

    # an LLM can reason over this. 
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
    # this creates the actual state object. 
    # a state is basically a snapshot your app 

    current_screen: str = "home"
    notes: List[str] = field(default_factory=list)
    focus_mode: bool = False
    notifications: bool = True
    username: str = STATIC_DATA["username"]
    email: str = STATIC_DATA["email"]
    app_version: str = STATIC_DATA["app_version"]

    # Transient / episode state
    note_input_buffer: str = ""
    # imagine typing "hello", temporarily, the note_input_buffer = "hello", after save it, notes=["hello"] and note_input_buffer=""

    invalid_action_count: int = 0
    # suppose an agent does, tap save_note_button while being on the home screen, that action is invalid. and hence, invalid_action_count += 1

    safety_violation: bool = False
    # it's for unsafe actions, like agent trying to delete all the data, maybe environment marks True. 

    steps_taken: int = 0
    # it counts, how many actions agent performed in this episode. 

    finished: bool = False
    # like the goal was, "create a note with title of xyz"
    # once agent completes, notes=['xyz'], finished = True. 


    def get_observation(self) -> str:
        # this converts internal state into something the agent sees. 

        # example
        # current_screen = "settings"
        # focus_mode = True
        # notifications = False

        # then the output becomes
        # "you are on the settings screen. 
        # Focus mode : True, Notifications : False, App version : 2.4.1"

        template = SCREEN_OBSERVATIONS[self.current_screen]
        return template.format(
            app_version=self.app_version,
            focus_mode=self.focus_mode,
            notifications=self.notifications,
            username=self.username,
            email=self.email,
        )

    def available_elements(self) -> set[str]:
        # returns available buttons, example screen="notes", the returns {add_note_button, note_input, save_note_button, back_button}
        return SCREEN_ELEMENTS[self.current_screen]

    def clone(self) -> "AppState":
        # this is important, as it creates a copy of the state but why???
        # RL algorithms often simulate, what happens if i take this action? 

        # current : state A, notes = [] 
        # try action, save_note_button
        # new state B, notes = ['hello']
        #  
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