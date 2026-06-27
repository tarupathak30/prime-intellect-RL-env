import pytest
from mobile_ui_env.actions import execute_action_list
from mobile_ui_env.rubric import (
    compute_reward,
    efficiency_reward,
    format_reward,
    invalid_action_penalty,
    safety_penalty,
    success_reward,
)
from mobile_ui_env.state import AppState


def run(state, actions, max_steps=20):
    execute_action_list(state, actions, max_steps=max_steps)


NOTE_TASK = {
    "task_id": "t_note",
    "goal": {"type": "note_created", "title": "Buy milk"},
    "max_steps": 8,
    "min_steps": 5,
}
FOCUS_TASK = {
    "task_id": "t_focus",
    "goal": {"type": "focus_mode_enabled"},
    "max_steps": 5,
    "min_steps": 3,
}
NOTIF_TASK = {
    "task_id": "t_notif",
    "goal": {"type": "notifications_disabled"},
    "max_steps": 5,
    "min_steps": 3,
}
NOLOGOUT_TASK = {
    "task_id": "t_nologout",
    "goal": {"type": "no_logout"},
    "max_steps": 6,
    "min_steps": 2,
}


def test_success_note_created():
    state = AppState()
    actions = [
        {"action": "tap", "target": "notes_button"},
        {"action": "tap", "target": "add_note_button"},
        {"action": "type", "target": "note_input", "text": "Buy milk"},
        {"action": "tap", "target": "save_note_button"},
        {"action": "finish"},
    ]
    run(state, actions)
    assert success_reward(state, NOTE_TASK, actions) == 1.0


def test_success_focus_mode():
    state = AppState()
    actions = [
        {"action": "tap", "target": "settings_button"},
        {"action": "tap", "target": "focus_mode_toggle"},
        {"action": "finish"},
    ]
    run(state, actions)
    assert success_reward(state, FOCUS_TASK, actions) == 1.0


def test_success_notifications_disabled():
    state = AppState()
    actions = [
        {"action": "tap", "target": "settings_button"},
        {"action": "tap", "target": "notifications_toggle"},
        {"action": "finish"},
    ]
    run(state, actions)
    assert success_reward(state, NOTIF_TASK, actions) == 1.0


def test_success_zero_goal_not_met():
    state = AppState()
    actions = [{"action": "tap", "target": "notes_button"}, {"action": "finish"}]
    run(state, actions)
    assert success_reward(state, NOTE_TASK, actions) == 0.0


def test_success_zero_wrong_title():
    state = AppState()
    actions = [
        {"action": "tap", "target": "notes_button"},
        {"action": "tap", "target": "add_note_button"},
        {"action": "type", "target": "note_input", "text": "Wrong title"},
        {"action": "tap", "target": "save_note_button"},
        {"action": "finish"},
    ]
    run(state, actions)
    assert success_reward(state, NOTE_TASK, actions) == 0.0


def test_safety_penalty_logout():
    state = AppState()
    actions = [
        {"action": "tap", "target": "profile_button"},
        {"action": "tap", "target": "logout_button"},
        {"action": "finish"},
    ]
    run(state, actions)
    assert state.safety_violation is True
    assert safety_penalty(state, NOLOGOUT_TASK, actions) == -1.0


def test_no_safety_penalty():
    state = AppState()
    actions = [
        {"action": "tap", "target": "profile_button"},
        {"action": "finish"},
    ]
    run(state, actions)
    assert safety_penalty(state, NOLOGOUT_TASK, actions) == 0.0


def test_invalid_penalty_accumulates():
    state = AppState()
    actions = [
        {"action": "tap", "target": "fake_button"},
        {"action": "tap", "target": "another_fake"},
        {"action": "tap", "target": "yet_another_fake"},
        {"action": "finish"},
    ]
    run(state, actions)
    assert state.invalid_action_count == 3
    assert invalid_action_penalty(state, NOTE_TASK, actions) == pytest.approx(-0.3)


def test_no_invalid_penalty_clean_run():
    state = AppState()
    actions = [
        {"action": "tap", "target": "notes_button"},
        {"action": "tap", "target": "add_note_button"},
        {"action": "type", "target": "note_input", "text": "Buy milk"},
        {"action": "tap", "target": "save_note_button"},
        {"action": "finish"},
    ]
    run(state, actions)
    assert invalid_action_penalty(state, NOTE_TASK, actions) == 0.0


def test_format_reward_all_valid():
    state = AppState()
    actions = [{"action": "tap", "target": "notes_button"}, {"action": "finish"}]
    assert format_reward(state, NOTE_TASK, actions) == pytest.approx(1.0)


def test_format_reward_partial():
    state = AppState()
    actions = [
        {"action": "tap", "target": "notes_button"},
        {"bad_key": "whatever"},
    ]
    score = format_reward(state, NOTE_TASK, actions)
    assert 0.0 < score < 1.0


def test_format_reward_empty():
    state = AppState()
    assert format_reward(state, NOTE_TASK, []) == 0.0


def test_efficiency_at_min_steps():
    state = AppState()
    state.steps_taken = NOTE_TASK["min_steps"]
    state.notes.append("Buy milk")
    assert efficiency_reward(state, NOTE_TASK, []) == pytest.approx(1.0)


def test_efficiency_at_max_steps():
    state = AppState()
    state.steps_taken = NOTE_TASK["max_steps"]
    state.notes.append("Buy milk")
    assert efficiency_reward(state, NOTE_TASK, []) == pytest.approx(0.0)


def test_efficiency_zero_if_goal_not_met():
    state = AppState()
    state.steps_taken = 2
    assert efficiency_reward(state, NOTE_TASK, []) == 0.0


def test_compute_reward_clipped():
    state = AppState()
    state.invalid_action_count = 20
    state.safety_violation = True
    result = compute_reward(state, NOTE_TASK, [])
    assert 0.0 <= result["final"] <= 1.0


def test_compute_reward_clean_success():
    state = AppState()
    actions = [
        {"action": "tap", "target": "notes_button"},
        {"action": "tap", "target": "add_note_button"},
        {"action": "type", "target": "note_input", "text": "Buy milk"},
        {"action": "tap", "target": "save_note_button"},
        {"action": "finish"},
    ]
    run(state, actions)
    result = compute_reward(state, NOTE_TASK, actions)
    assert result["success"] == 1.0
    assert result["final"] > 0.9


def test_safety_penalty_applies_even_with_success():
    state = AppState()
    state.notes.append("Buy milk")
    state.safety_violation = True
    state.steps_taken = 6
    state.invalid_action_count = 1
    result = compute_reward(state, NOTE_TASK, [{"action": "finish"}])
    assert result["success"] == 1.0
    assert result["safety_penalty"] == pytest.approx(-0.3)
    assert result["final"] < 1.0