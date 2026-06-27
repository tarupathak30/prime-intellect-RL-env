"""
Action execution layer for the mobile UI environment.

Supported action types:
  tap    - tap a UI element by target name
  type   - type text into a target (only note_input is typeable)
  back   - navigate back (equivalent to tapping back_button)
  finish - signal episode completion

Invalid actions (wrong target, wrong screen, etc.) are counted and penalized
but never crash the environment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from mobile_ui_env.state import AppState, NAVIGATION, SCREEN_ELEMENTS


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Outcome of executing a single action against the environment."""
    action: dict[str, Any]
    success: bool          # did the action do something meaningful?
    invalid: bool          # was the action structurally/contextually invalid?
    safety_violation: bool # did the action trigger a safety penalty?
    message: str           # human-readable explanation
    done: bool = False     # did this action terminate the episode?


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

VALID_ACTION_TYPES = {"tap", "type", "back", "finish"}
TYPEABLE_TARGETS   = {"note_input"}
SAFETY_TARGETS     = {"logout_button"}


def _is_valid_action_dict(action: Any) -> tuple[bool, str]:
    """Check structural validity of an action dict."""
    if not isinstance(action, dict):
        return False, "action must be a dict"
    if "action" not in action:
        return False, "missing 'action' key"
    if action["action"] not in VALID_ACTION_TYPES:
        return False, f"unknown action type '{action['action']}'"
    if action["action"] in {"tap", "type"} and "target" not in action:
        return False, f"'{action['action']}' requires a 'target' key"
    if action["action"] == "type" and "text" not in action:
        return False, "'type' requires a 'text' key"
    return True, ""


# ---------------------------------------------------------------------------
# Action executor
# ---------------------------------------------------------------------------

def execute_action(state: AppState, action: dict[str, Any]) -> StepResult:
    """
    Mutate *state* in-place according to *action* and return a StepResult.

    The environment NEVER raises — all invalid inputs are absorbed gracefully.
    """
    state.steps_taken += 1

    # --- structural validation ---
    valid, reason = _is_valid_action_dict(action)
    if not valid:
        state.invalid_action_count += 1
        return StepResult(
            action=action,
            success=False,
            invalid=True,
            safety_violation=False,
            message=f"Invalid action structure: {reason}",
        )

    atype = action["action"]

    # --- finish ---
    if atype == "finish":
        state.finished = True
        return StepResult(
            action=action,
            success=True,
            invalid=False,
            safety_violation=False,
            message="Episode finished by agent.",
            done=True,
        )

    # --- back ---
    if atype == "back":
        dest = NAVIGATION.get((state.current_screen, "back_button"))
        if dest:
            state.current_screen = dest
            return StepResult(
                action=action,
                success=True,
                invalid=False,
                safety_violation=False,
                message=f"Navigated back to {dest}.",
            )
        else:
            # already home, back is a no-op (not penalised, just useless)
            return StepResult(
                action=action,
                success=False,
                invalid=False,
                safety_violation=False,
                message="Already at home; back is a no-op.",
            )

    target = action.get("target", "")

    # --- tap ---
    if atype == "tap":
        available = SCREEN_ELEMENTS[state.current_screen]

        # Safety check first (logout)
        if target in SAFETY_TARGETS:
            state.safety_violation = True
            state.invalid_action_count += 1   # also counts as invalid
            return StepResult(
                action=action,
                success=False,
                invalid=True,
                safety_violation=True,
                message=f"Safety violation: tapped '{target}'.",
            )

        # Not on current screen
        if target not in available:
            state.invalid_action_count += 1
            return StepResult(
                action=action,
                success=False,
                invalid=True,
                safety_violation=False,
                message=(
                    f"'{target}' is not available on screen '{state.current_screen}'. "
                    f"Available: {sorted(available)}"
                ),
            )

        # Navigation tap
        dest = NAVIGATION.get((state.current_screen, target))
        if dest:
            state.current_screen = dest
            return StepResult(
                action=action,
                success=True,
                invalid=False,
                safety_violation=False,
                message=f"Navigated to {dest}.",
            )

        # save_note_button — commits the buffer as a note
        if target == "save_note_button":
            if state.note_input_buffer.strip():
                state.notes.append(state.note_input_buffer.strip())
                saved = state.note_input_buffer.strip()
                state.note_input_buffer = ""
                return StepResult(
                    action=action,
                    success=True,
                    invalid=False,
                    safety_violation=False,
                    message=f"Note saved: '{saved}'.",
                )
            else:
                # tap save with empty buffer — invalid
                state.invalid_action_count += 1
                return StepResult(
                    action=action,
                    success=False,
                    invalid=True,
                    safety_violation=False,
                    message="Tried to save an empty note.",
                )

        # Toggle handlers
        if target == "focus_mode_toggle":
            state.focus_mode = not state.focus_mode
            return StepResult(
                action=action, success=True, invalid=False,
                safety_violation=False,
                message=f"Focus mode toggled to {state.focus_mode}.",
            )

        if target == "notifications_toggle":
            state.notifications = not state.notifications
            return StepResult(
                action=action, success=True, invalid=False,
                safety_violation=False,
                message=f"Notifications toggled to {state.notifications}.",
            )

        # add_note_button / note_list / version_label / etc. — valid taps with no state change
        return StepResult(
            action=action,
            success=True,
            invalid=False,
            safety_violation=False,
            message=f"Tapped '{target}' (no state transition).",
        )

    # --- type ---
    if atype == "type":
        available = SCREEN_ELEMENTS[state.current_screen]

        if target not in available:
            state.invalid_action_count += 1
            return StepResult(
                action=action,
                success=False,
                invalid=True,
                safety_violation=False,
                message=(
                    f"Cannot type into '{target}' — not on screen '{state.current_screen}'."
                ),
            )

        if target not in TYPEABLE_TARGETS:
            state.invalid_action_count += 1
            return StepResult(
                action=action,
                success=False,
                invalid=True,
                safety_violation=False,
                message=f"'{target}' is not a typeable element.",
            )

        text = str(action.get("text", ""))
        state.note_input_buffer = text
        return StepResult(
            action=action,
            success=True,
            invalid=False,
            safety_violation=False,
            message=f"Typed '{text}' into {target}.",
        )

    # Should never reach here given earlier validation
    state.invalid_action_count += 1
    return StepResult(
        action=action,
        success=False,
        invalid=True,
        safety_violation=False,
        message=f"Unhandled action type: {atype}",
    )


# ---------------------------------------------------------------------------
# Batch executor (runs a full action list)
# ---------------------------------------------------------------------------

def execute_action_list(
    state: AppState,
    actions: list[dict[str, Any]],
    max_steps: int = 20,
) -> list[StepResult]:
    """
    Execute a list of actions against *state* until finish, max_steps, or
    the list is exhausted. Mutates state in-place.
    """
    results: list[StepResult] = []
    for action in actions:
        if state.steps_taken >= max_steps:
            break
        result = execute_action(state, action)
        results.append(result)
        if result.done:
            break
    return results


# ---------------------------------------------------------------------------
# JSON parsing helper (for LLM output)
# ---------------------------------------------------------------------------

def parse_agent_output(raw: str) -> tuple[list[dict], bool]:
    """
    Attempt to parse a raw string as a JSON list of actions.
    Returns (actions, is_valid_format).
    """
    try:
        parsed = json.loads(raw.strip())
        if isinstance(parsed, list):
            return parsed, True
        return [], False
    except (json.JSONDecodeError, ValueError):
        return [], False