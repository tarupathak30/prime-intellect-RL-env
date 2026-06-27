import pytest
from mobile_ui_env.environment import MobileUIEnv, load_environment
from mobile_ui_env.dataset import get_task


@pytest.fixture
def note_task():
    return get_task("task_001")

@pytest.fixture
def nologout_task():
    return get_task("task_013")


def test_reset_returns_observation(note_task):
    env = MobileUIEnv(note_task)
    obs = env.reset()
    assert isinstance(obs, str)
    assert len(obs) > 0
    assert "home" in obs.lower()


def test_step_returns_correct_types(note_task):
    env = MobileUIEnv(note_task)
    env.reset()
    obs, reward, done, info = env.step([{"action": "tap", "target": "notes_button"}])
    assert isinstance(obs, str)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_step_reward_bounded(note_task):
    env = MobileUIEnv(note_task)
    env.reset()
    actions = [
        {"action": "tap", "target": "notes_button"},
        {"action": "tap", "target": "add_note_button"},
        {"action": "type", "target": "note_input", "text": "Buy milk"},
        {"action": "tap", "target": "save_note_button"},
        {"action": "finish"},
    ]
    _, reward, _, _ = env.step(actions)
    assert 0.0 <= reward <= 1.0


def test_step_info_keys(note_task):
    env = MobileUIEnv(note_task)
    env.reset()
    _, _, _, info = env.step([{"action": "finish"}])
    for key in ("steps_taken", "invalid_actions", "safety_violation", "current_screen", "notes", "reward_breakdown"):
        assert key in info


def test_episode_ends_at_max_steps(note_task):
    env = MobileUIEnv(note_task)
    env.reset()
    _, _, done, info = env.step([{"action": "tap", "target": "notes_button"}] * 20)
    assert done is True
    assert info["steps_taken"] <= note_task["max_steps"]


def test_episode_ends_on_finish(note_task):
    env = MobileUIEnv(note_task)
    env.reset()
    _, _, done, info = env.step([{"action": "finish"}])
    assert done is True
    assert info["finished_by_agent"] is True


def test_step_after_done(note_task):
    env = MobileUIEnv(note_task)
    env.reset()
    env.step([{"action": "finish"}])
    _, _, done, info = env.step([{"action": "tap", "target": "notes_button"}])
    assert done is True
    assert "error" in info


def test_logout_sets_safety_violation(nologout_task):
    env = MobileUIEnv(nologout_task)
    env.reset()
    _, _, done, info = env.step([
        {"action": "tap", "target": "profile_button"},
        {"action": "tap", "target": "logout_button"},
        {"action": "finish"},
    ])
    assert info["safety_violation"] is True
    assert done is True


def test_logout_reduces_reward(nologout_task):
    env = MobileUIEnv(nologout_task)
    env.reset()
    _, clean, _, _ = env.step([
        {"action": "tap", "target": "profile_button"},
        {"action": "finish"},
    ])

    env2 = MobileUIEnv(nologout_task)
    env2.reset()
    _, bad, _, _ = env2.step([
        {"action": "tap", "target": "profile_button"},
        {"action": "tap", "target": "logout_button"},
        {"action": "finish"},
    ])

    assert clean > bad


def test_load_environment_datasets(note_task):
    env = load_environment()
    assert hasattr(env, "dataset")
    assert hasattr(env, "eval_dataset")
    assert len(env.dataset) == 20
    assert len(env.eval_dataset) == 10


def test_load_environment_splits():
    env = load_environment()
    for task in env.dataset:
        assert task["split"] == "train"
    for task in env.eval_dataset:
        assert task["split"] == "eval"


def test_run_episode(note_task):
    shim = load_environment()
    result = shim.run_episode(note_task, [
        {"action": "tap", "target": "notes_button"},
        {"action": "tap", "target": "add_note_button"},
        {"action": "type", "target": "note_input", "text": "Buy milk"},
        {"action": "tap", "target": "save_note_button"},
        {"action": "finish"},
    ])
    assert result["reward"] > 0.9


def test_state_resets(note_task):
    env = MobileUIEnv(note_task)
    env.reset()
    env.step([
        {"action": "tap", "target": "notes_button"},
        {"action": "tap", "target": "add_note_button"},
        {"action": "type", "target": "note_input", "text": "Buy milk"},
        {"action": "tap", "target": "save_note_button"},
        {"action": "finish"},
    ])
    assert len(env.state.notes) == 1

    env.reset()
    assert len(env.state.notes) == 0
    assert env.state.current_screen == "home"
    assert env.done is False