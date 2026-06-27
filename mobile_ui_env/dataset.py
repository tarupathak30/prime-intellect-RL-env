"""
Dataset builder for the mobile UI environment.

30 tasks total:
  - 20 train tasks (task_001 … task_020)
  - 10 eval tasks  (task_021 … task_030)

Each task has:
  task_id      : unique string ID
  instruction  : natural language instruction for the agent
  goal         : structured goal dict (checked by rubric)
  max_steps    : hard episode limit
  min_steps    : optimal (minimum) step count for efficiency scoring
  split        : "train" | "eval"
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Raw task definitions
# ---------------------------------------------------------------------------

_RAW_TASKS: list[dict[str, Any]] = [
    # ── TRAIN ──────────────────────────────────────────────────────────────

    # Note creation (single)
    {
        "task_id": "task_001",
        "instruction": 'Create a note titled "Buy milk"',
        "goal": {"type": "note_created", "title": "Buy milk"},
        "max_steps": 8,
        "min_steps": 5,
    },
    {
        "task_id": "task_002",
        "instruction": 'Create a note titled "Call dentist"',
        "goal": {"type": "note_created", "title": "Call dentist"},
        "max_steps": 8,
        "min_steps": 5,
    },
    {
        "task_id": "task_003",
        "instruction": 'Add a note that says "Team standup at 10am"',
        "goal": {"type": "note_created", "title": "Team standup at 10am"},
        "max_steps": 8,
        "min_steps": 5,
    },
    {
        "task_id": "task_004",
        "instruction": 'Save a note with the text "Pick up dry cleaning"',
        "goal": {"type": "note_created", "title": "Pick up dry cleaning"},
        "max_steps": 8,
        "min_steps": 5,
    },
    {
        "task_id": "task_005",
        "instruction": 'Create a note titled "Review PR before EOD"',
        "goal": {"type": "note_created", "title": "Review PR before EOD"},
        "max_steps": 8,
        "min_steps": 5,
    },

    # Settings — focus mode
    {
        "task_id": "task_006",
        "instruction": "Enable focus mode",
        "goal": {"type": "focus_mode_enabled"},
        "max_steps": 5,
        "min_steps": 3,
    },
    {
        "task_id": "task_007",
        "instruction": "Turn off focus mode",
        "goal": {"type": "focus_mode_disabled"},
        "max_steps": 5,
        "min_steps": 3,
    },
    {
        "task_id": "task_008",
        "instruction": "Go to settings and enable focus mode",
        "goal": {"type": "focus_mode_enabled"},
        "max_steps": 6,
        "min_steps": 3,
    },

    # Settings — notifications
    {
        "task_id": "task_009",
        "instruction": "Disable notifications",
        "goal": {"type": "notifications_disabled"},
        "max_steps": 5,
        "min_steps": 3,
    },
    {
        "task_id": "task_010",
        "instruction": "Turn on notifications",
        "goal": {"type": "notifications_enabled"},
        "max_steps": 5,
        "min_steps": 3,
    },

    # Profile — read info
    {
        "task_id": "task_011",
        "instruction": "Find the username from the profile screen",
        "goal": {"type": "read_username"},
        "max_steps": 5,
        "min_steps": 2,
    },
    {
        "task_id": "task_012",
        "instruction": "Find the email address from the profile screen",
        "goal": {"type": "read_email"},
        "max_steps": 5,
        "min_steps": 2,
    },
    {
        "task_id": "task_013",
        "instruction": "Go to the profile screen and do NOT tap logout",
        "goal": {"type": "no_logout"},
        "max_steps": 6,
        "min_steps": 2,
    },

    # Settings — read version
    {
        "task_id": "task_014",
        "instruction": "Open settings and find the app version",
        "goal": {"type": "read_version"},
        "max_steps": 5,
        "min_steps": 2,
    },

    # Two notes
    {
        "task_id": "task_015",
        "instruction": 'Create two notes: "Buy milk" and "Walk the dog"',
        "goal": {"type": "notes_created", "titles": ["Buy milk", "Walk the dog"]},
        "max_steps": 14,
        "min_steps": 9,
    },
    {
        "task_id": "task_016",
        "instruction": 'Add notes "Morning run" and "Evening yoga"',
        "goal": {"type": "notes_created", "titles": ["Morning run", "Evening yoga"]},
        "max_steps": 14,
        "min_steps": 9,
    },

    # Navigate to a specific screen
    {
        "task_id": "task_017",
        "instruction": "Navigate to the Notes screen",
        "goal": {"type": "screen_visited", "screen": "notes"},
        "max_steps": 4,
        "min_steps": 1,
    },
    {
        "task_id": "task_018",
        "instruction": "Navigate to the Settings screen",
        "goal": {"type": "screen_visited", "screen": "settings"},
        "max_steps": 4,
        "min_steps": 1,
    },

    # Composite: note + settings
    {
        "task_id": "task_019",
        "instruction": 'Create a note "Grocery list" and then enable focus mode',
        "goal": {
            "type": "composite",
            "subgoals": [
                {"type": "note_created", "title": "Grocery list"},
                {"type": "focus_mode_enabled"},
            ],
        },
        "max_steps": 16,
        "min_steps": 9,
    },
    {
        "task_id": "task_020",
        "instruction": "Disable notifications and then go to the profile screen",
        "goal": {
            "type": "composite",
            "subgoals": [
                {"type": "notifications_disabled"},
                {"type": "screen_visited", "screen": "profile"},
            ],
        },
        "max_steps": 10,
        "min_steps": 5,
    },

    # ── EVAL ───────────────────────────────────────────────────────────────

    {
        "task_id": "task_021",
        "instruction": 'Create a note titled "Dentist appointment at 3pm"',
        "goal": {"type": "note_created", "title": "Dentist appointment at 3pm"},
        "max_steps": 8,
        "min_steps": 5,
    },
    {
        "task_id": "task_022",
        "instruction": "Go to the profile screen and read the email address without logging out",
        "goal": {
            "type": "composite",
            "subgoals": [
                {"type": "read_email"},
                {"type": "no_logout"},
            ],
        },
        "max_steps": 6,
        "min_steps": 2,
    },
    {
        "task_id": "task_023",
        "instruction": "Enable focus mode in settings",
        "goal": {"type": "focus_mode_enabled"},
        "max_steps": 5,
        "min_steps": 3,
    },
    {
        "task_id": "task_024",
        "instruction": 'Create two notes: "Sprint planning" and "Retrospective"',
        "goal": {"type": "notes_created", "titles": ["Sprint planning", "Retrospective"]},
        "max_steps": 14,
        "min_steps": 9,
    },
    {
        "task_id": "task_025",
        "instruction": "Open settings and check the app version label",
        "goal": {"type": "read_version"},
        "max_steps": 5,
        "min_steps": 2,
    },
    {
        "task_id": "task_026",
        "instruction": "Disable notifications",
        "goal": {"type": "notifications_disabled"},
        "max_steps": 5,
        "min_steps": 3,
    },
    {
        "task_id": "task_027",
        "instruction": 'Save a note with the text "Fix login bug"',
        "goal": {"type": "note_created", "title": "Fix login bug"},
        "max_steps": 8,
        "min_steps": 5,
    },
    {
        "task_id": "task_028",
        "instruction": "Navigate to the profile screen and find the username",
        "goal": {"type": "read_username"},
        "max_steps": 5,
        "min_steps": 2,
    },
    {
        "task_id": "task_029",
        "instruction": 'Create a note "Water plants" and then disable notifications',
        "goal": {
            "type": "composite",
            "subgoals": [
                {"type": "note_created", "title": "Water plants"},
                {"type": "notifications_disabled"},
            ],
        },
        "max_steps": 16,
        "min_steps": 9,
    },
    {
        "task_id": "task_030",
        "instruction": "Go to the profile screen and do NOT press logout",
        "goal": {"type": "no_logout"},
        "max_steps": 6,
        "min_steps": 2,
    },
]


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------

_SPLIT_MAP: dict[str, str] = {}
for _t in _RAW_TASKS:
    tid = _t["task_id"]
    num = int(tid.split("_")[1])
    _SPLIT_MAP[tid] = "train" if num <= 20 else "eval"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dataset(split: str = "train") -> list[dict[str, Any]]:
    """
    Return task dicts for the requested split ("train" or "eval").

    Each returned dict has:
      task_id, instruction, goal, max_steps, min_steps, split
    """
    if split not in {"train", "eval"}:
        raise ValueError(f"Unknown split '{split}'. Choose 'train' or 'eval'.")

    result = []
    for task in _RAW_TASKS:
        s = _SPLIT_MAP[task["task_id"]]
        if s == split:
            result.append({**task, "split": s})
    return result


def get_task(task_id: str) -> dict[str, Any]:
    """Look up a single task by ID."""
    for task in _RAW_TASKS:
        if task["task_id"] == task_id:
            return {**task, "split": _SPLIT_MAP[task["task_id"]]}
    raise KeyError(f"Task '{task_id}' not found.")


def all_tasks() -> list[dict[str, Any]]:
    """Return all 30 tasks across both splits."""
    return [{**t, "split": _SPLIT_MAP[t["task_id"]]} for t in _RAW_TASKS]