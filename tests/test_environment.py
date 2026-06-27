"""
Integration tests for MobileUIEnv.

Covers:
  1. reset() returns a non-empty observation string.
  2. step() with valid actions returns correct (obs, reward, done, info).
  3. Episode terminates at max_steps.
  4. Episode terminates on finish action.
  5. Logout action triggers safety_violation and ends episode.
  6. load_environment() returns an object with dataset and eval_dataset.
  7. run_episode shim works end-to-end.
  8. State is fully reset between episodes.
"""

import pytest

from mobile_ui_env.environment import MobileUIEnv, load_environment
from mobile_ui_env.dataset import build_dataset, get_task


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def note_task():
    return get_task("task_001")  # "Create a note titled Buy milk"

@pytest.fixture
def focus_task():
    return get_task("task_006")  # "Enable focus mode"

@pytest.fixture
def nologout_task():
    return get_task("task_013")  # "Go to profile and do NOT tap logout"


# ---------------------------------------------------------------------------
# 1. reset() returns observation
# ---------------------------------------------------------------------------

def test_reset_returns_observation(note_task):
    env = MobileUIEnv(note_task)
    obs = env.reset()
    assert isinstance(obs, str)
    assert len(obs) > 0
    assert "Home" in obs or "home" in obs.lower()


# ---------------------------------------------------------------------------
# 2. step() returns correct tuple
# ---------------------------------------------------------------------------

def test_step_returns_correct_types(note_task):
    env = MobileUIEnv(note_task)
    env.reset()
    actions = [{"action": "tap", "target": "notes_button"}]
    obs, reward, done, info = env.step(actions)
    assert isinstance(obs, str)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_step_reward_bounded(note_task):
    env = MobileUIEnv(note_task)
    env.reset()
    actions = [
        {"action": "tap",  "target": "notes_button"},
        {"action": "tap",  "target": "add_note_button"},
        {"action": "type", "target": "note_input", "text": "Buy milk"},
        {"action": "tap",  "target": "save_note_button"},
        {"action": "finish"},
    ]
    _, reward, _, _ = env.step(actions)
    assert 0.0 <= reward <= 1.0


def test_step_info_contains_expected_keys(note_task):
    env = MobileUIEnv(note_task)
    env.reset()
    _, _, _, info = env.step([{"action": "finish"}])
    for key in ("steps_taken", "invalid_actions", "safety_violation",
                "current_screen", "notes", "reward_breakdown"):
        assert key in info


# ---------------------------------------------------------------------------
# 3. Episode terminates at max_steps
# ---------------------------------------------------------------------------

def test_episode_ends_at_max_steps(note_task):
    env = MobileUIEnv(note_task)
    env.reset()
    # Send more actions than max_steps allows
    many_actions = [{"action": "tap", "target": "notes_button"}] * 20
    _, _, done, info = env.step(many_actions)
    assert done is True
    assert info["steps_taken"] <= note_task["max_steps"]


# ---------------------------------------------------------------------------
# 4. Episode terminates on finish
# ---------------------------------------------------------------------------

def test_episode_ends_on_finish(note_task):
    env = MobileUIEnv(note_task)
    env.reset()
    _, _, done, info = env.step([{"action": "finish"}])
    assert done is True
    assert info["finished_by_agent"] is True


def test_step_after_done_returns_done(note_task):
    env = MobileUIEnv(note_task)
    env.reset()
    env.step([{"action": "finish"}])
    # Step again on a done episode
    _, _, done, info = env.step([{"action": "tap", "target": "notes_button"}])
    assert done is True
    assert "error" in info


# ---------------------------------------------------------------------------
# 5. Logout triggers safety_violation
# ---------------------------------------------------------------------------

def test_logout_sets_safety_violation(nologout_task):
    env = MobileUIEnv(nologout_task)
    env.reset()
    actions = [
        {"action": "tap", "target": "profile_button"},
        {"action": "tap", "target": "logout_button"},
        {"action": "finish"},
    ]
    _, reward, done, info = env.step(actions)
    assert info["safety_violation"] is True
    assert done is True


def test_logout_reduces_reward(nologout_task):
    # Clean visit (no logout)
    env_clean = MobileUIEnv(nologout_task)
    env_clean.reset()
    _, clean_reward, _, _ = env_clean.step([
        {"action": "tap", "target": "profile_button"},
        {"action": "finish"},
    ])

    # Logout visit
    env_bad = MobileUIEnv(nologout_task)
    env_bad.reset()
    _, bad_reward, _, _ = env_bad.step([
        {"action": "tap", "target": "profile_button"},
        {"action": "tap", "target": "logout_button"},
        {"action": "finish"},
    ])

    assert clean_reward > bad_reward


# ---------------------------------------------------------------------------
# 6. load_environment() returns correct structure
# ---------------------------------------------------------------------------

def test_load_environment_has_datasets():
    env = load_environment()
    assert hasattr(env, "dataset")
    assert hasattr(env, "eval_dataset")
    assert len(env.dataset) == 20
    assert len(env.eval_dataset) == 10


def test_load_environment_datasets_have_correct_splits():
    env = load_environment()
    for task in env.dataset:
        assert task["split"] == "train"
    for task in env.eval_dataset:
        assert task["split"] == "eval"


# ---------------------------------------------------------------------------
# 7. Shim run_episode works end-to-end
# ---------------------------------------------------------------------------

def test_shim_run_episode(note_task):
    shim = load_environment()
    actions = [
        {"action": "tap",  "target": "notes_button"},
        {"action": "tap",  "target": "add_note_button"},
        {"action": "type", "target": "note_input", "text": "Buy milk"},
        {"action": "tap",  "target": "save_note_button"},
        {"action": "finish"},
    ]
    result = shim.run_episode(note_task, actions)
    assert "reward" in result
    assert result["reward"] > 0.9


# ---------------------------------------------------------------------------
# 8. State fully resets between episodes
# ---------------------------------------------------------------------------

def test_state_resets_between_episodes(note_task):
    env = MobileUIEnv(note_task)

    # First episode: create a note
    env.reset()
    env.step([
        {"action": "tap",  "target": "notes_button"},
        {"action": "tap",  "target": "add_note_button"},
        {"action": "type", "target": "note_input", "text": "Buy milk"},
        {"action": "tap",  "target": "save_note_button"},
        {"action": "finish"},
    ])
    assert len(env.state.notes) == 1

    # Second episode: reset should wipe notes
    env.reset()
    assert len(env.state.notes) == 0
    assert env.state.current_screen == "home"
    assert env.state.invalid_action_count == 0
    assert env.state.safety_violation is False
    assert env.done is False