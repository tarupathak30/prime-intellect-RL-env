import pytest
from mobile_ui_env.actions import execute_action, execute_action_list, parse_agent_output
from mobile_ui_env.state import AppState


@pytest.fixture
def state():
    return AppState()


def test_tap_notes_button(state):
    execute_action(state, {"action": "tap", "target": "notes_button"})
    assert state.current_screen == "notes"


def test_tap_settings_button(state):
    execute_action(state, {"action": "tap", "target": "settings_button"})
    assert state.current_screen == "settings"


def test_tap_profile_button(state):
    execute_action(state, {"action": "tap", "target": "profile_button"})
    assert state.current_screen == "profile"


def test_invalid_tap_doesnt_crash(state):
    result = execute_action(state, {"action": "tap", "target": "nonexistent_button"})
    assert result.invalid is True
    assert state.current_screen == "home"
    assert state.invalid_action_count == 1


def test_tap_element_from_wrong_screen(state):
    # add_note_button is on notes screen, not home
    result = execute_action(state, {"action": "tap", "target": "add_note_button"})
    assert result.invalid is True


def test_invalid_taps_accumulate(state):
    for _ in range(4):
        execute_action(state, {"action": "tap", "target": "doesnt_exist"})
    assert state.invalid_action_count == 4


def test_missing_action_key(state):
    result = execute_action(state, {"target": "notes_button"})
    assert result.invalid is True


def test_bad_action_type(state):
    result = execute_action(state, {"action": "swipe", "target": "notes_button"})
    assert result.invalid is True


def test_note_creation(state):
    execute_action(state, {"action": "tap", "target": "notes_button"})
    execute_action(state, {"action": "tap", "target": "add_note_button"})
    execute_action(state, {"action": "type", "target": "note_input", "text": "Buy milk"})
    execute_action(state, {"action": "tap", "target": "save_note_button"})
    assert "Buy milk" in state.notes


def test_save_empty_note_is_invalid(state):
    execute_action(state, {"action": "tap", "target": "notes_button"})
    result = execute_action(state, {"action": "tap", "target": "save_note_button"})
    assert result.invalid is True
    assert len(state.notes) == 0


def test_two_notes(state):
    execute_action(state, {"action": "tap", "target": "notes_button"})
    execute_action(state, {"action": "tap", "target": "add_note_button"})
    execute_action(state, {"action": "type", "target": "note_input", "text": "Note A"})
    execute_action(state, {"action": "tap", "target": "save_note_button"})
    execute_action(state, {"action": "tap", "target": "add_note_button"})
    execute_action(state, {"action": "type", "target": "note_input", "text": "Note B"})
    execute_action(state, {"action": "tap", "target": "save_note_button"})
    assert len(state.notes) == 2
    assert "Note A" in state.notes and "Note B" in state.notes


def test_type_into_non_typeable(state):
    execute_action(state, {"action": "tap", "target": "notes_button"})
    result = execute_action(state, {"action": "type", "target": "save_note_button", "text": "oops"})
    assert result.invalid is True


def test_back_from_notes(state):
    execute_action(state, {"action": "tap", "target": "notes_button"})
    execute_action(state, {"action": "back"})
    assert state.current_screen == "home"


def test_back_from_settings(state):
    execute_action(state, {"action": "tap", "target": "settings_button"})
    execute_action(state, {"action": "back"})
    assert state.current_screen == "home"


def test_finish(state):
    result = execute_action(state, {"action": "finish"})
    assert result.done is True
    assert state.finished is True


def test_max_steps_respected(state):
    actions = [{"action": "tap", "target": "notes_button"}] * 20
    execute_action_list(state, actions, max_steps=3)
    assert state.steps_taken <= 3


def test_stops_at_finish(state):
    actions = [
        {"action": "tap", "target": "notes_button"},
        {"action": "finish"},
        {"action": "tap", "target": "settings_button"},  # shouldn't run
    ]
    results = execute_action_list(state, actions, max_steps=10)
    assert state.current_screen == "notes"
    assert any(r.done for r in results)


def test_parse_valid_json(state):
    raw = '[{"action": "tap", "target": "notes_button"}, {"action": "finish"}]'
    actions, ok = parse_agent_output(raw)
    assert ok is True
    assert len(actions) == 2


def test_parse_bad_json(state):
    actions, ok = parse_agent_output("not json")
    assert ok is False
    assert actions == []


def test_parse_object_not_list(state):
    actions, ok = parse_agent_output('{"action": "tap"}')
    assert ok is False


def test_parse_empty_list(state):
    actions, ok = parse_agent_output("[]")
    assert ok is True