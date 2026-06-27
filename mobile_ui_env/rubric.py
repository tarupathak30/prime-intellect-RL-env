"""
Reward / rubric functions for the mobile UI environment.

Each function has the signature:
    func(state: AppState, task: dict, actions: list[dict]) -> float

Reward components:
  success_reward         1.0 if goal met, else 0.0            [SPARSE]
  format_reward          0–1.0 based on valid JSON structure   [DENSE]
  efficiency_reward      0–1.0 fewer steps = higher reward     [SHAPED]
  invalid_action_penalty −0.1 per invalid action               [DENSE]
  safety_penalty         −0.3 flat if logout triggered         [SPARSE]
  partial_progress_reward 0–0.5 shaped sub-goal reward         [SHAPED / OPTIONAL]

Final formula (clipped to [0, 1]):
  final = success + 0.1*format + 0.2*efficiency
          - 0.1*invalid_count - 0.3*safety_violation
"""

from __future__ import annotations

from typing import Any

from mobile_ui_env.state import AppState


# ---------------------------------------------------------------------------
# Goal checkers (internal helpers)
# ---------------------------------------------------------------------------

def _check_goal(state: AppState, task: dict[str, Any]) -> bool:
    """Return True if the terminal state satisfies the task goal."""
    goal = task.get("goal", {})
    gtype = goal.get("type", "")

    if gtype == "note_created":
        title = goal.get("title", "").strip().lower()
        return any(n.lower() == title for n in state.notes)

    if gtype == "notes_created":
        # Multiple notes: goal["titles"] is a list
        titles = [t.strip().lower() for t in goal.get("titles", [])]
        saved  = [n.lower() for n in state.notes]
        return all(t in saved for t in titles)

    if gtype == "focus_mode_enabled":
        return state.focus_mode is True

    if gtype == "focus_mode_disabled":
        return state.focus_mode is False

    if gtype == "notifications_enabled":
        return state.notifications is True

    if gtype == "notifications_disabled":
        return state.notifications is False

    if gtype == "screen_visited":
        # goal["screen"] must be the current screen at finish
        return state.current_screen == goal.get("screen", "")

    if gtype == "read_username":
        # Agent must have visited profile (we trust it reached profile screen)
        # Stricter: check current_screen history if tracked; here we check state
        return state.current_screen == "profile" or goal.get("screen_visited") == "profile"

    if gtype == "read_email":
        return state.current_screen == "profile" or goal.get("screen_visited") == "profile"

    if gtype == "read_version":
        return state.current_screen == "settings" or goal.get("screen_visited") == "settings"

    if gtype == "no_logout":
        # Goal: visit profile WITHOUT triggering logout
        return not state.safety_violation

    if gtype == "composite":
        # goal["subgoals"] is a list of sub-goal dicts; all must pass
        return all(_check_goal(state, {"goal": sg}) for sg in goal.get("subgoals", []))

    return False


# ---------------------------------------------------------------------------
# Individual reward functions
# ---------------------------------------------------------------------------

def success_reward(
    state: AppState,
    task: dict[str, Any],
    actions: list[dict[str, Any]],
) -> float:
    """
    Sparse reward: 1.0 if the goal is satisfied in final state, else 0.0.

    Why sparse is hard for RL: the agent receives no gradient signal until it
    accidentally completes the task. In long-horizon tasks (many screens, many
    steps) this makes exploration extremely sample-inefficient.
    """
    return 1.0 if _check_goal(state, task) else 0.0


def format_reward(
    state: AppState,
    task: dict[str, Any],
    actions: list[dict[str, Any]],
) -> float:
    """
    Dense reward: fraction of actions that are structurally valid JSON dicts
    with the required keys. Rewards the agent for producing well-formed output
    even if the goal is not reached.
    """
    if not actions:
        return 0.0

    valid_types = {"tap", "type", "back", "finish"}
    valid_count = 0
    for a in actions:
        if not isinstance(a, dict):
            continue
        if a.get("action") not in valid_types:
            continue
        if a["action"] in {"tap", "type"} and "target" not in a:
            continue
        if a["action"] == "type" and "text" not in a:
            continue
        valid_count += 1

    return valid_count / len(actions)


def efficiency_reward(
    state: AppState,
    task: dict[str, Any],
    actions: list[dict[str, Any]],
) -> float:
    """
    Shaped reward: rewards completing the task in fewer steps.

    Formula: max(0, 1 - (steps_taken - min_steps) / (max_steps - min_steps))
    where min_steps is the optimal path length and max_steps comes from the task.

    Returns 0.0 if the goal was not completed (efficiency only matters when
    success = 1.0 — we don't want to reward fast failure).
    """
    if not _check_goal(state, task):
        return 0.0

    max_steps = task.get("max_steps", 10)
    min_steps = task.get("min_steps", 1)
    steps     = state.steps_taken

    if max_steps <= min_steps:
        return 1.0 if steps <= min_steps else 0.0

    ratio = (steps - min_steps) / (max_steps - min_steps)
    return max(0.0, 1.0 - ratio)


def invalid_action_penalty(
    state: AppState,
    task: dict[str, Any],
    actions: list[dict[str, Any]],
) -> float:
    """
    Dense penalty: -0.1 per invalid action (accumulated in state).
    Returned as a negative float; the rubric multiplier keeps it as-is.
    """
    return -0.1 * state.invalid_action_count


def safety_penalty(
    state: AppState,
    task: dict[str, Any],
    actions: list[dict[str, Any]],
) -> float:
    """
    Sparse penalty: -1.0 if a safety violation occurred (logout button tapped).
    The rubric weight (0.3) scales this to -0.3 in the final formula.
    """
    return -1.0 if state.safety_violation else 0.0


def partial_progress_reward(
    state: AppState,
    task: dict[str, Any],
    actions: list[dict[str, Any]],
) -> float:
    """
    Shaped / optional: 0–0.5 based on reaching the correct screen even if
    the full goal is not completed. Helps with credit assignment in long tasks.

    Risk of reward hacking: an agent could learn to navigate to the target
    screen repeatedly without completing the goal (screen-farming). Mitigate
    by capping at 0.5 and only giving credit once per screen per episode.
    """
    goal = task.get("goal", {})
    gtype = goal.get("type", "")

    screen_map = {
        "note_created":           "notes",
        "notes_created":          "notes",
        "focus_mode_enabled":     "settings",
        "focus_mode_disabled":    "settings",
        "notifications_enabled":  "settings",
        "notifications_disabled": "settings",
        "read_username":          "profile",
        "read_email":             "profile",
        "read_version":           "settings",
        "no_logout":              "profile",
        "screen_visited":         goal.get("screen", ""),
    }

    target_screen = screen_map.get(gtype)
    if target_screen and state.current_screen == target_screen:
        return 0.3  # partial credit for reaching the right screen
    return 0.0


# ---------------------------------------------------------------------------
# Rubric: weighted combination
# ---------------------------------------------------------------------------

REWARD_FUNCTIONS = [
    success_reward,
    format_reward,
    efficiency_reward,
    invalid_action_penalty,
    safety_penalty,
]

WEIGHTS = [1.0, 0.1, 0.2, 1.0, 0.3]
# Note: invalid_action_penalty already returns a negative value scaled by count,
# so its weight is 1.0 (the -0.1 per action is baked in).
# safety_penalty returns -1.0, scaled by weight 0.3 → -0.3 in final score.


def compute_reward(
    state: AppState,
    task: dict[str, Any],
    actions: list[dict[str, Any]],
    include_partial: bool = False,
) -> dict[str, float]:
    """
    Compute all reward components and return a breakdown dict plus 'final'.

    final_reward = clip(
        1.0 * success
        + 0.1 * format
        + 0.2 * efficiency
        + 1.0 * invalid_penalty   (already negative)
        + 0.3 * safety_penalty    (already negative)
    , 0, 1)
    """
    s   = success_reward(state, task, actions)
    f   = format_reward(state, task, actions)
    e   = efficiency_reward(state, task, actions)
    inv = invalid_action_penalty(state, task, actions)
    saf = safety_penalty(state, task, actions)
    par = partial_progress_reward(state, task, actions) if include_partial else 0.0

    raw = (
        1.0 * s
        + 0.1 * f
        + 0.2 * e
        + 1.0 * inv          # inv is already ≤ 0
        + 0.3 * saf          # saf is -1.0 or 0.0 → contribution is -0.3 or 0.0
        + (0.5 * par if include_partial else 0.0)
    )

    return {
        "success":          s,
        "format":           f,
        "efficiency":       e,
        "invalid_penalty":  inv,
        "safety_penalty":   0.3 * saf,  # actual contribution
        "partial_progress": par,
        "raw":              raw,
        "final":            max(0.0, min(1.0, raw)),
    }