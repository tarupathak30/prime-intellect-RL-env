"""
MobileUIEnv: the main RL-style environment class.

Episode lifecycle:
  env = MobileUIEnv(task)
  obs  = env.reset()                   # returns initial observation string
  obs, reward, done, info = env.step(actions)  # accepts list of action dicts
  # done when: agent calls finish, max_steps exceeded, or safety violation

Termination conditions:
  1. Agent emits {"action": "finish"}
  2. steps_taken >= task["max_steps"]
  3. safety_violation (logout tapped) — episode ends immediately with penalty

load_environment() returns a Verifiers-compatible object (or a plain dict
if the verifiers package is not installed).
"""

from __future__ import annotations

from typing import Any

from mobile_ui_env.actions import execute_action_list, parse_agent_output
from mobile_ui_env.dataset import build_dataset
from mobile_ui_env.rubric import compute_reward
from mobile_ui_env.state import AppState


# ---------------------------------------------------------------------------
# Core environment class
# ---------------------------------------------------------------------------

class MobileUIEnv:
    """
    Single-episode RL environment for a mobile UI agent.

    Parameters
    ----------
    task : dict
        A task dict as returned by build_dataset() or get_task().
    include_partial : bool
        Whether to include partial_progress_reward in the final score.
    """

    def __init__(self, task: dict[str, Any], include_partial: bool = False):
        self.task = task
        self.include_partial = include_partial
        self.state: AppState = AppState()
        self._done: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> str:
        """Reset episode state and return the initial observation."""
        self.state = AppState()
        self._done = False
        return self.state.get_observation()

    def step(
        self,
        actions: list[dict[str, Any]] | str,
    ) -> tuple[str, float, bool, dict[str, Any]]:
        """
        Execute *actions* (list of dicts or raw JSON string) and return
        (observation, reward, done, info).

        reward is the *incremental* reward for this step batch (for multi-turn
        use cases). For single-turn evaluation use compute_full_reward().
        """
        if self._done:
            return (
                self.state.get_observation(),
                0.0,
                True,
                {"error": "Episode already done. Call reset()."},
            )

        # Parse if raw string
        if isinstance(actions, str):
            actions, fmt_ok = parse_agent_output(actions)
            if not fmt_ok:
                self.state.invalid_action_count += 1

        max_steps = self.task.get("max_steps", 20)
        results = execute_action_list(self.state, actions, max_steps=max_steps)

        # Episode termination
        finished_by_agent = any(r.done for r in results)
        over_limit = self.state.steps_taken >= max_steps
        safety_hit = self.state.safety_violation

        self._done = finished_by_agent or over_limit or safety_hit

        reward_breakdown = compute_reward(
            self.state, self.task, actions, include_partial=self.include_partial
        )

        info = {
            "steps_taken":       self.state.steps_taken,
            "invalid_actions":   self.state.invalid_action_count,
            "safety_violation":  self.state.safety_violation,
            "current_screen":    self.state.current_screen,
            "notes":             list(self.state.notes),
            "finished_by_agent": finished_by_agent,
            "over_limit":        over_limit,
            "reward_breakdown":  reward_breakdown,
            "step_results":      [
                {"action": r.action, "success": r.success,
                 "invalid": r.invalid, "message": r.message}
                for r in results
            ],
        }

        return (
            self.state.get_observation(),
            reward_breakdown["final"],
            self._done,
            info,
        )

    def compute_full_reward(
        self,
        actions: list[dict[str, Any]],
    ) -> dict[str, float]:
        """
        Convenience method: reset, run all actions, return full reward breakdown.
        Does NOT mutate long-term state (resets first).
        """
        self.reset()
        self.step(actions)
        return compute_reward(
            self.state, self.task, actions, include_partial=self.include_partial
        )

    @property
    def done(self) -> bool:
        return self._done

    @property
    def observation(self) -> str:
        return self.state.get_observation()


# ---------------------------------------------------------------------------
# Verifiers-compatible load_environment()
# ---------------------------------------------------------------------------

def load_environment(include_partial: bool = False) -> Any:
    """
    Return a Verifiers-compatible environment object.

    If the `verifiers` package is installed, returns vf.SingleTurnEnv.
    Otherwise returns a lightweight dict-based shim with the same interface
    so the rest of the codebase works without the dependency.
    """
    from mobile_ui_env.dataset import build_dataset
    from mobile_ui_env.rubric import (
        success_reward,
        format_reward,
        efficiency_reward,
        invalid_action_penalty,
        safety_penalty,
    )

    dataset      = build_dataset(split="train")
    eval_dataset = build_dataset(split="eval")

    try:
        import verifiers as vf  # type: ignore

        rubric = vf.Rubric(
            funcs=[
                success_reward,
                format_reward,
                efficiency_reward,
                invalid_action_penalty,
                safety_penalty,
            ],
            weights=[1.0, 0.1, 0.2, 1.0, 0.3],
        )

        return vf.SingleTurnEnv(
            dataset=dataset,
            eval_dataset=eval_dataset,
            rubric=rubric,
        )

    except ImportError:
        # Lightweight shim — exposes .dataset, .eval_dataset, .make(task)
        class _EnvShim:
            def __init__(self):
                self.dataset      = dataset
                self.eval_dataset = eval_dataset

            def make(self, task: dict[str, Any]) -> MobileUIEnv:
                """Instantiate a single-task environment."""
                return MobileUIEnv(task, include_partial=include_partial)

            def run_episode(
                self,
                task: dict[str, Any],
                actions: list[dict[str, Any]],
            ) -> dict[str, Any]:
                """Run one episode and return reward breakdown + info."""
                env = self.make(task)
                env.reset()
                obs, reward, done, info = env.step(actions)
                return {"reward": reward, "done": done, **info}

            def __repr__(self):
                return (
                    f"<MobileUIEnvShim  train={len(self.dataset)} tasks  "
                    f"eval={len(self.eval_dataset)} tasks>"
                )

        return _EnvShim()