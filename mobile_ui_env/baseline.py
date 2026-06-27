"""
baseline.py — heuristic agent that generates action lists for tasks.

This is NOT a trained model. It uses rule-based logic derived from the task
goal type to produce near-optimal action sequences. Used in run_eval.py to
establish a performance ceiling and verify the environment works end-to-end.
"""

from __future__ import annotations

from typing import Any


def heuristic_agent(task: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Given a task dict, return a list of actions that attempt to complete it.
    Covers all goal types in the dataset.
    """
    goal  = task.get("goal", {})
    gtype = goal.get("type", "")

    if gtype == "note_created":
        return _make_single_note(goal["title"])

    if gtype == "notes_created":
        titles = goal.get("titles", [])
        if not titles:
            return [{"action": "finish"}]
        # Navigate to notes screen once
        actions: list[dict] = [{"action": "tap", "target": "notes_button"}]
        for title in titles:
            actions += [
                {"action": "tap",  "target": "add_note_button"},
                {"action": "type", "target": "note_input", "text": title},
                {"action": "tap",  "target": "save_note_button"},
            ]
        actions.append({"action": "finish"})
        return actions

    if gtype == "focus_mode_enabled":
        return [
            {"action": "tap", "target": "settings_button"},
            {"action": "tap", "target": "focus_mode_toggle"},
            {"action": "finish"},
        ]

    if gtype == "focus_mode_disabled":
        return [
            {"action": "tap", "target": "settings_button"},
            {"action": "tap", "target": "focus_mode_toggle"},
            {"action": "finish"},
        ]

    if gtype == "notifications_enabled":
        return [
            {"action": "tap", "target": "settings_button"},
            {"action": "tap", "target": "notifications_toggle"},
            {"action": "finish"},
        ]

    if gtype == "notifications_disabled":
        return [
            {"action": "tap", "target": "settings_button"},
            {"action": "tap", "target": "notifications_toggle"},
            {"action": "finish"},
        ]

    if gtype == "read_username":
        return [
            {"action": "tap", "target": "profile_button"},
            {"action": "tap", "target": "username_label"},
            {"action": "finish"},
        ]

    if gtype == "read_email":
        return [
            {"action": "tap", "target": "profile_button"},
            {"action": "tap", "target": "email_label"},
            {"action": "finish"},
        ]

    if gtype == "read_version":
        return [
            {"action": "tap", "target": "settings_button"},
            {"action": "tap", "target": "version_label"},
            {"action": "finish"},
        ]

    if gtype == "no_logout":
        # Visit profile, observe, do NOT tap logout
        return [
            {"action": "tap", "target": "profile_button"},
            {"action": "finish"},
        ]

    if gtype == "screen_visited":
        screen = goal.get("screen", "home")
        button_map = {
            "notes":    "notes_button",
            "settings": "settings_button",
            "profile":  "profile_button",
            "home":     None,
        }
        btn = button_map.get(screen)
        if btn:
            return [{"action": "tap", "target": btn}, {"action": "finish"}]
        return [{"action": "finish"}]

    if gtype == "composite":
        actions: list[dict] = []
        for subgoal in goal.get("subgoals", []):
            sub_actions = heuristic_agent({"goal": subgoal, "max_steps": 20, "min_steps": 1})
            # Remove "finish" from all but the last subgoal
            sub_actions = [a for a in sub_actions if a.get("action") != "finish"]
            # If we need to go back to home between subgoals, add back navigation
            actions += sub_actions
            actions.append({"action": "back"})  # return home between subgoals
        # Remove last unnecessary back, add finish
        if actions and actions[-1].get("action") == "back":
            actions.pop()
        actions.append({"action": "finish"})
        return actions

    # Fallback: do nothing useful
    return [{"action": "finish"}]


def _make_single_note(title: str) -> list[dict[str, Any]]:
    """Return the canonical action sequence to create one note."""
    return [
        {"action": "tap",  "target": "notes_button"},
        {"action": "tap",  "target": "add_note_button"},
        {"action": "type", "target": "note_input", "text": title},
        {"action": "tap",  "target": "save_note_button"},
        {"action": "finish"},
    ]


def dummy_agent(task: dict[str, Any]) -> list[dict[str, Any]]:
    """
    A deliberately bad agent that always taps the wrong things.
    Useful for verifying that invalid actions are penalised correctly.
    """
    return [
        {"action": "tap", "target": "nonexistent_button"},
        {"action": "type", "target": "settings_button", "text": "oops"},
        {"action": "finish"},
    ]


def random_agent(task: dict[str, Any], seed: int = 42) -> list[dict[str, Any]]:
    """
    Agent that picks random valid actions from the current screen.
    For stress-testing the environment's crash-resistance.
    """
    import random
    from mobile_ui_env.state import SCREEN_ELEMENTS

    rng = random.Random(seed)
    actions: list[dict] = []
    screen = "home"
    max_steps = task.get("max_steps", 10)

    nav_map = {
        ("home", "notes_button"):    "notes",
        ("home", "settings_button"): "settings",
        ("home", "profile_button"):  "profile",
        ("notes", "back_button"):    "home",
        ("settings", "back_button"): "home",
        ("profile", "back_button"):  "home",
    }

    for _ in range(max_steps - 1):
        elements = list(SCREEN_ELEMENTS[screen])
        target = rng.choice(elements)
        actions.append({"action": "tap", "target": target})
        screen = nav_map.get((screen, target), screen)

    actions.append({"action": "finish"})
    return actions