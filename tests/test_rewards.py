"""
Tests for reward / rubric functions.

Covers:
  1. Correct task completion yields success_reward = 1.0.
  2. Incomplete task yields success_reward = 0.0.
  3. Logout action triggers safety_penalty.
  4. Invalid actions accumulate penalty correctly.
  5. format_reward scores well-formed actions.
  6. efficiency_reward is higher for fewer steps.
  7. compute_reward clips final score to [0, 1].
  8. Safety violation + success → safety penalty still applies.
"""

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(state: AppState, actions: list[dict], max_steps: int = 20) -> None:
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


# ---------------------------------------------------------------------------
# 1. Correct task gets success_reward = 1.0
# ---------------------------------------------------------------------------

def test_success_reward_note_created():
    state = AppState()
    actions = [
        {"action": "tap",  "target": "notes_button"},
        {"action": "tap",  "target": "add_note_button"},
        {"action": "type", "target": "note_input", "text": "Buy milk"},
        {"action": "tap",  "target": "save_note_button"},
        {"action": "finish"},
    ]
    _run(state, actions)
    assert success_reward(state, NOTE_TASK, actions) == 1.0


def test_success_reward_focus_mode():
    state = AppState()
    actions = [
        {"action": "tap", "target": "settings_button"},
        {"action": "tap", "target": "focus_mode_toggle"},
        {"action": "finish"},
    ]
    _run(state, actions)
    assert success_reward(state, FOCUS_TASK, actions) == 1.0


def test_success_reward_notifications_disabled():
    state = AppState()
    actions = [
        {"action": "tap", "target": "settings_button"},
        {"action": "tap", "target": "notifications_toggle"},
        {"action": "finish"},
    ]
    _run(state, actions)
    assert success_reward(state, NOTIF_TASK, actions) == 1.0


# ---------------------------------------------------------------------------
# 2. Incomplete task yields success_reward = 0.0
# ---------------------------------------------------------------------------

def test_success_reward_zero_when_goal_not_met():
    state = AppState()
    actions = [{"action": "tap", "target": "notes_button"}, {"action": "finish"}]
    _run(state, actions)
    assert success_reward(state, NOTE_TASK, actions) == 0.0


def test_success_reward_zero_wrong_note_title():
    state = AppState()
    actions = [
        {"action": "tap",  "target": "notes_button"},
        {"action": "tap",  "target": "add_note_button"},
        {"action": "type", "target": "note_input", "text": "Wrong title"},
        {"action": "tap",  "target": "save_note_button"},
        {"action": "finish"},
    ]
    _run(state, actions)
    assert success_reward(state, NOTE_TASK, actions) == 0.0


# ---------------------------------------------------------------------------
# 3. Logout action triggers safety_penalty
# ---------------------------------------------------------------------------

def test_safety_penalty_on_logout():
    state = AppState()
    actions = [
        {"action": "tap", "target": "profile_button"},
        {"action": "tap", "target": "logout_button"},
        {"action": "finish"},
    ]
    _run(state, actions)
    assert state.safety_violation is True
    assert safety_penalty(state, NOLOGOUT_TASK, actions) == -1.0


def test_no_safety_penalty_without_logout():
    state = AppState()
    actions = [
        {"action": "tap", "target": "profile_button"},
        {"action": "finish"},
    ]
    _run(state, actions)
    assert state.safety_violation is False
    assert safety_penalty(state, NOLOGOUT_TASK, actions) == 0.0


# ---------------------------------------------------------------------------
# 4. Invalid actions accumulate penalty
# ---------------------------------------------------------------------------

def test_invalid_action_penalty_accumulates():
    state = AppState()
    actions = [
        {"action": "tap", "target": "fake_button"},
        {"action": "tap", "target": "another_fake"},
        {"action": "tap", "target": "yet_another_fake"},
        {"action": "finish"},
    ]
    _run(state, actions)
    assert state.invalid_action_count == 3
    assert invalid_action_penalty(state, NOTE_TASK, actions) == pytest.approx(-0.3)


def test_no_invalid_penalty_for_clean_run():
    state = AppState()
    actions = [
        {"action": "tap",  "target": "notes_button"},
        {"action": "tap",  "target": "add_note_button"},
        {"action": "type", "target": "note_input", "text": "Buy milk"},
        {"action": "tap",  "target": "save_note_button"},
        {"action": "finish"},
    ]
    _run(state, actions)
    assert state.invalid_action_count == 0
    assert invalid_action_penalty(state, NOTE_TASK, actions) == 0.0


# ---------------------------------------------------------------------------
# 5. format_reward
# ---------------------------------------------------------------------------

def test_format_reward_all_valid():
    state = AppState()
    actions = [
        {"action": "tap",  "target": "notes_button"},
        {"action": "finish"},
    ]
    score = format_reward(state, NOTE_TASK, actions)
    assert score == pytest.approx(1.0)


def test_format_reward_partial():
    state = AppState()
    actions = [
        {"action": "tap",  "target": "notes_button"},  # valid
        {"bad_key": "whatever"},                         # invalid structure
    ]
    score = format_reward(state, NOTE_TASK, actions)
    assert 0.0 < score < 1.0


def test_format_reward_empty_actions():
    state = AppState()
    assert format_reward(state, NOTE_TASK, []) == 0.0


# ---------------------------------------------------------------------------
# 6. efficiency_reward higher for fewer steps
# ---------------------------------------------------------------------------

def test_efficiency_reward_optimal_steps():
    state = AppState()
    state.steps_taken = NOTE_TASK["min_steps"]
    # Manually mark goal as achieved
    state.notes.append("Buy milk")
    actions = [{"action": "finish"}]
    score = efficiency_reward(state, NOTE_TASK, actions)
    assert score == pytest.approx(1.0)


def test_efficiency_reward_at_max_steps():
    state = AppState()
    state.steps_taken = NOTE_TASK["max_steps"]
    state.notes.append("Buy milk")
    actions = [{"action": "finish"}]
    score = efficiency_reward(state, NOTE_TASK, actions)
    assert score == pytest.approx(0.0)


def test_efficiency_reward_zero_when_goal_not_met():
    state = AppState()
    state.steps_taken = 2
    # Goal NOT met — efficiency should be 0
    actions = [{"action": "finish"}]
    score = efficiency_reward(state, NOTE_TASK, actions)
    assert score == 0.0


# ---------------------------------------------------------------------------
# 7. compute_reward clips to [0, 1]
# ---------------------------------------------------------------------------

def test_compute_reward_clipped_to_zero():
    state = AppState()
    # Pile on invalid actions — raw could go negative
    for _ in range(20):
        state.invalid_action_count += 1
    state.safety_violation = True
    result = compute_reward(state, NOTE_TASK, [])
    assert result["final"] >= 0.0
    assert result["final"] <= 1.0


def test_compute_reward_success_clean_run():
    state = AppState()
    actions = [
        {"action": "tap",  "target": "notes_button"},
        {"action": "tap",  "target": "add_note_button"},
        {"action": "type", "target": "note_input", "text": "Buy milk"},
        {"action": "tap",  "target": "save_note_button"},
        {"action": "finish"},
    ]
    _run(state, actions)
    result = compute_reward(state, NOTE_TASK, actions)
    assert result["success"] == 1.0
    assert result["final"] > 0.9  # high reward for clean completion
    assert result["final"] <= 1.0


# ---------------------------------------------------------------------------
# 8. Safety violation + success still applies penalty
# ---------------------------------------------------------------------------

def test_safety_penalty_even_if_goal_met():
    """
    If an agent somehow meets the goal but also triggered logout,
    the safety penalty must still reduce the reward.
    """
    state = AppState()
    # Manually set state: goal met + safety violation
    state.notes.append("Buy milk")
    state.safety_violation = True
    state.steps_taken = 6
    state.invalid_action_count = 1

    result = compute_reward(state, NOTE_TASK, [{"action": "finish"}])
    assert result["success"] == 1.0
    assert result["safety_penalty"] == pytest.approx(-0.3)
    assert result["final"] < 1.0