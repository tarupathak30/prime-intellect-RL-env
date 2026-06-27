from __future__ import annotations

from typing import Any

from mobile_ui_env.actions import execute_action_list, parse_agent_output
from mobile_ui_env.dataset import build_dataset
from mobile_ui_env.state import AppState
from mobile_ui_env.rubric import (
    success_reward,
    format_reward,
    efficiency_reward,
    invalid_action_penalty,
    safety_penalty,
    compute_reward
)
# this file is basically the brain/controller of mobile UI reinforcement learning environment. 
# the RL loop goes like : agent chooses actions, MobileUIEnv.step(actions) modifies app state, then apptate calculates reward, reward function return observation + reward + done. 

# that's how agent runs.


# Core environment class


class MobileUIEnv:
    
    def __init__(self, task: dict[str, Any], include_partial: bool = False):
        self.task = task #store a task 
        self.include_partial = include_partial
        self.state: AppState = AppState() #create initial state 
        self._done: bool = False #episode status


    # Public API
    

    def reset(self) -> str:
        # as in RL, every episode starts fresh.  
        self.state = AppState() #destroy old world and create new world.
        self._done = False #allow agent to take actions again. 
        return self.state.get_observation() #return what agent sees. 

    def step(
        self,
        actions: list[dict[str, Any]] | str,
    ) -> tuple[str, float, bool, dict[str, Any]]:
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
            if not fmt_ok: #in case formatting is wrong.
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
            "steps_taken":self.state.steps_taken,
            "invalid_actions":self.state.invalid_action_count,
            "safety_violation":self.state.safety_violation,
            "current_screen":self.state.current_screen,
            "notes":list(self.state.notes),
            "finished_by_agent": finished_by_agent,
            "over_limit":over_limit,
            "reward_breakdown": reward_breakdown,
            "step_results":[
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


# Verifiers-compatible load_environment()

def load_environment(include_partial: bool = False) -> Any:
    dataset = build_dataset(split="train")
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
        # Lightweight shim : exposes .dataset, .eval_dataset, .make(task)
        class _EnvShim:
            def __init__(self):
                self.dataset      = dataset
                self.eval_dataset = eval_dataset

            def make(self, task: dict[str, Any]) -> MobileUIEnv:
                return MobileUIEnv(task, include_partial=include_partial)

            def run_episode(
                self,
                task: dict[str, Any],
                actions: list[dict[str, Any]],
            ) -> dict[str, Any]:
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