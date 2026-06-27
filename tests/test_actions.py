"""
Tests for action execution logic.

Covers:
  1. Valid tap changes screen.
  2. Invalid tap does not crash the environment.
  3. Creating a note updates state.
  4. Typing into a non-typeable element is penalised.
  5. Back navigation returns to home.
  6. Finish action sets done flag.
  7. Action list respects max_steps.
  8. parse_agent_output handles valid and invalid JSON.
"""

import pytest

from mobile_ui_env.actions import (
    execute_action,
    execute_action_list,
    parse_agent_output,
)
from mobile_ui_env.state import AppState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_state():
    return AppState()


# ---------------------------------------------------------------------------
# 1. Valid tap changes screen
# ---------------------------------------------------------------------------

def test_tap_notes_button_changes_screen(fresh_state):
    result = execute_action(fresh_state, {"action": "tap", "target": "notes_button"})
    assert result.success is True
    assert result.invalid is False
    assert fresh_state.current_screen == "notes"


def test_tap_settings_button_changes_screen(fresh_state):
    execute_action(fresh_state, {"action": "tap", "target": "settings_button"})
    assert fresh_state.current_screen == "settings"


def test_tap_profile_button_changes_screen(fresh_state):
    execute_action(fresh_state, {"action": "tap", "target": "profile_button"})
    assert fresh_state.current_screen == "profile"


# ---------------------------------------------------------------------------
# 2. Invalid tap does NOT crash the environment
# ---------------------------------------------------------------------------

def test_invalid_tap_does_not_raise(fresh_state):
    # Should not raise — just penalise
    result = execute_action(fresh_state, {"action": "tap", "target": "nonexistent_button"})
    assert result.invalid is True
    assert fresh_state.current_screen == "home"  # no navigation happened
    assert fresh_state.invalid_action_count == 1


def test_tap_wrong_screen_element_does_not_raise(fresh_state):
    # add_note_button only exists on notes screen, not home
    result = execute_action(fresh_state, {"action": "tap", "target": "add_note_button"})
    assert result.invalid is True
    assert fresh_state.invalid_action_count == 1


def test_multiple_invalid_taps_accumulate(fresh_state):
    for _ in range(5):
        execute_action(fresh_state, {"action": "tap", "target": "bad_button"})
    assert fresh_state.invalid_action_count == 5


def test_missing_action_key_is_invalid(fresh_state):
    result = execute_action(fresh_state, {"target": "notes_button"})
    assert result.invalid is True


def test_unknown_action_type_is_invalid(fresh_state):
    result = execute_action(fresh_state, {"action": "swipe", "target": "notes_button"})
    assert result.invalid is True


# ---------------------------------------------------------------------------
# 3. Creating a note updates state
# ---------------------------------------------------------------------------

def test_full_note_creation_flow(fresh_state):
    execute_action(fresh_state, {"action": "tap",  "target": "notes_button"})
    execute_action(fresh_state, {"action": "tap",  "target": "add_note_button"})
    execute_action(fresh_state, {"action": "type", "target": "note_input", "text": "Buy milk"})
    execute_action(fresh_state, {"action": "tap",  "target": "save_note_button"})

    assert "Buy milk" in fresh_state.notes
    assert len(fresh_state.notes) == 1


def test_save_without_type_is_invalid(fresh_state):
    execute_action(fresh_state, {"action": "tap", "target": "notes_button"})
    result = execute_action(fresh_state, {"action": "tap", "target": "save_note_button"})
    assert result.invalid is True
    assert len(fresh_state.notes) == 0


def test_two_notes_created(fresh_state):
    execute_action(fresh_state, {"action": "tap",  "target": "notes_button"})
    execute_action(fresh_state, {"action": "tap",  "target": "add_note_button"})
    execute_action(fresh_state, {"action": "type", "target": "note_input", "text": "Note A"})
    execute_action(fresh_state, {"action": "tap",  "target": "save_note_button"})
    execute_action(fresh_state, {"action": "tap",  "target": "add_note_button"})
    execute_action(fresh_state, {"action": "type", "target": "note_input", "text": "Note B"})
    execute_action(fresh_state, {"action": "tap",  "target": "save_note_button"})

    assert "Note A" in fresh_state.notes
    assert "Note B" in fresh_state.notes
    assert len(fresh_state.notes) == 2


# ---------------------------------------------------------------------------
# 4. Typing into a non-typeable element is penalised
# ---------------------------------------------------------------------------

def test_type_into_non_typeable_target(fresh_state):
    execute_action(fresh_state, {"action": "tap", "target": "notes_button"})
    # save_note_button is tappable but not typeable
    result = execute_action(
        fresh_state,
        {"action": "type", "target": "save_note_button", "text": "oops"},
    )
    assert result.invalid is True
    assert fresh_state.invalid_action_count == 1


# ---------------------------------------------------------------------------
# 5. Back navigation returns to home
# ---------------------------------------------------------------------------

def test_back_from_notes_returns_home(fresh_state):
    execute_action(fresh_state, {"action": "tap", "target": "notes_button"})
    assert fresh_state.current_screen == "notes"
    execute_action(fresh_state, {"action": "back"})
    assert fresh_state.current_screen == "home"


def test_back_from_settings_returns_home(fresh_state):
    execute_action(fresh_state, {"action": "tap", "target": "settings_button"})
    execute_action(fresh_state, {"action": "back"})
    assert fresh_state.current_screen == "home"


# ---------------------------------------------------------------------------
# 6. Finish action sets done
# ---------------------------------------------------------------------------

def test_finish_action_sets_done(fresh_state):
    result = execute_action(fresh_state, {"action": "finish"})
    assert result.done is True
    assert fresh_state.finished is True


# ---------------------------------------------------------------------------
# 7. Action list respects max_steps
# ---------------------------------------------------------------------------

def test_execute_action_list_respects_max_steps(fresh_state):
    actions = [{"action": "tap", "target": "notes_button"}] * 20
    results = execute_action_list(fresh_state, actions, max_steps=3)
    assert fresh_state.steps_taken <= 3


def test_execute_action_list_stops_at_finish(fresh_state):
    actions = [
        {"action": "tap", "target": "notes_button"},
        {"action": "finish"},
        {"action": "tap", "target": "settings_button"},  # should not execute
    ]
    results = execute_action_list(fresh_state, actions, max_steps=10)
    assert fresh_state.current_screen == "notes"  # never reached settings
    assert any(r.done for r in results)


# ---------------------------------------------------------------------------
# 8. parse_agent_output
# ---------------------------------------------------------------------------

def test_parse_valid_json_list():
    raw = '[{"action": "tap", "target": "notes_button"}, {"action": "finish"}]'
    actions, ok = parse_agent_output(raw)
    assert ok is True
    assert len(actions) == 2


def test_parse_invalid_json():
    actions, ok = parse_agent_output("not json at all")
    assert ok is False
    assert actions == []


def test_parse_json_object_not_list():
    actions, ok = parse_agent_output('{"action": "tap", "target": "x"}')
    assert ok is False


def test_parse_empty_list():
    actions, ok = parse_agent_output("[]")
    assert ok is True
    assert actions == []