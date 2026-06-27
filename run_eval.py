
from __future__ import annotations

import argparse
import json
import time
from typing import Any

from mobile_ui_env.actions import parse_agent_output
from mobile_ui_env.baseline import dummy_agent, heuristic_agent, random_agent
from mobile_ui_env.dataset import build_dataset
from mobile_ui_env.environment import MobileUIEnv


# ---------------------------------------------------------------------------
# LLM agent (optional — requires openai or compatible package)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a mobile UI agent. You control a simple mobile app with four screens:
- Home: notes_button, settings_button, profile_button
- Notes: add_note_button, note_input, save_note_button, note_list, back_button
- Settings: focus_mode_toggle, notifications_toggle, version_label, back_button
- Profile: username_label, email_label, logout_button, back_button

You start on the Home screen.

You must output a JSON array of actions. Each action is one of:
  {"action": "tap", "target": "<element>"}
  {"action": "type", "target": "note_input", "text": "<text>"}
  {"action": "back"}
  {"action": "finish"}

Output ONLY valid JSON. No explanation, no markdown fences.
"""


def llm_agent(task: dict[str, Any], base_url: str = "http://localhost:11434/v1",
              model: str = "llama3") -> list[dict[str, Any]]:
    """Call an OpenAI-compatible endpoint to get actions."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        print("  [LLM] openai package not installed. Falling back to heuristic.")
        return heuristic_agent(task)

    client = OpenAI(base_url=base_url, api_key="ollama")  # key ignored by local servers
    prompt = f"Task: {task['instruction']}\n\nOutput a JSON action list to complete this task."

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        raw = response.choices[0].message.content or "[]"
        actions, ok = parse_agent_output(raw)
        if not ok:
            print(f"  [LLM] Failed to parse output: {raw[:80]}")
            return [{"action": "finish"}]
        return actions
    except Exception as e:
        print(f"  [LLM] Error: {e}. Falling back to heuristic.")
        return heuristic_agent(task)


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_eval(
    split:    str  = "eval",
    agent:    str  = "heuristic",
    verbose:  bool = False,
    partial:  bool = False,
    llm_url:  str  = "http://localhost:11434/v1",
    llm_model: str = "llama3",
) -> dict[str, Any]:

    tasks = build_dataset(split=split)
    results: list[dict[str, Any]] = []

    agent_fn = {
        "heuristic": heuristic_agent,
        "dummy":     dummy_agent,
        "random":    random_agent,
    }.get(agent)

    print(f"\n{'='*60}")
    print(f"  Mobile UI Env — Evaluation")
    print(f"  Agent : {agent}   Split : {split}   Tasks : {len(tasks)}")
    if partial:
        print("  Partial progress reward: ENABLED")
    print(f"{'='*60}\n")

    for task in tasks:
        env = MobileUIEnv(task, include_partial=partial)
        env.reset()

        t0 = time.perf_counter()
        if agent_fn:
            actions = agent_fn(task)
        else:
            actions = llm_agent(task, base_url=llm_url, model=llm_model)

        _, reward, done, info = env.step(actions)
        elapsed = time.perf_counter() - t0

        breakdown = info["reward_breakdown"]
        row = {
            "task_id":        task["task_id"],
            "instruction":    task["instruction"],
            "success":        breakdown["success"] == 1.0,
            "reward":         breakdown["final"],
            "steps":          info["steps_taken"],
            "invalid":        info["invalid_actions"],
            "safety":         info["safety_violation"],
            "elapsed_s":      elapsed,
            "breakdown":      breakdown,
        }
        results.append(row)

        if verbose:
            status = "✓" if row["success"] else "✗"
            print(
                f"  [{status}] {task['task_id']:<12} "
                f"reward={row['reward']:.3f}  "
                f"steps={row['steps']}  "
                f"invalid={row['invalid']}  "
                f"safety={'YES' if row['safety'] else 'no'}"
            )
            if not row["success"]:
                print(f"       → {task['instruction']}")
            if verbose and row["invalid"] > 0:
                for sr in info.get("step_results", []):
                    if sr["invalid"]:
                        print(f"         ✗ {sr['action']} → {sr['message']}")

    # ── Aggregate metrics ──────────────────────────────────────────────────
    n               = len(results)
    successes       = sum(1 for r in results if r["success"])
    success_rate    = successes / n if n else 0.0
    avg_reward      = sum(r["reward"]  for r in results) / n if n else 0.0
    avg_steps       = sum(r["steps"]   for r in results) / n if n else 0.0
    avg_invalid     = sum(r["invalid"] for r in results) / n if n else 0.0
    safety_count    = sum(1 for r in results if r["safety"])
    invalid_rate    = avg_invalid / (avg_steps if avg_steps else 1)

    print(f"\n{'─'*60}")
    print(f"  Total {split} tasks  : {n}")
    print(f"  Success rate         : {success_rate*100:.1f}%  ({successes}/{n})")
    print(f"  Average reward       : {avg_reward:.4f}")
    print(f"  Average steps        : {avg_steps:.2f}")
    print(f"  Invalid action rate  : {invalid_rate:.4f}  (avg {avg_invalid:.2f} per episode)")
    print(f"  Safety violations    : {safety_count}")
    print(f"{'─'*60}\n")

    # Failure analysis
    failures = [r for r in results if not r["success"]]
    if failures:
        print(f"  ── Failure analysis ({len(failures)} tasks) ──")
        for r in failures:
            print(f"     {r['task_id']}: {r['instruction']}")
            bd = r["breakdown"]
            print(
                f"       success={bd['success']:.1f}  "
                f"efficiency={bd['efficiency']:.2f}  "
                f"invalid_pen={bd['invalid_penalty']:.2f}  "
                f"safety_pen={bd['safety_penalty']:.2f}"
            )
        print()

    metrics = {
        "split":             split,
        "agent":             agent,
        "n_tasks":           n,
        "successes":         successes,
        "success_rate":      success_rate,
        "avg_reward":        avg_reward,
        "avg_steps":         avg_steps,
        "avg_invalid":       avg_invalid,
        "invalid_rate":      invalid_rate,
        "safety_violations": safety_count,
        "per_task":          results,
    }
    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate the mobile UI environment.")
    parser.add_argument("--agent",     default="heuristic",
                        choices=["heuristic", "dummy", "random", "llm"],
                        help="Agent type to use")
    parser.add_argument("--split",     default="eval",
                        choices=["train", "eval"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--verbose",   action="store_true",
                        help="Print per-task results")
    parser.add_argument("--partial",   action="store_true",
                        help="Include partial_progress_reward")
    parser.add_argument("--llm-url",   default="http://localhost:11434/v1",
                        help="Base URL for OpenAI-compatible LLM server")
    parser.add_argument("--llm-model", default="llama3",
                        help="Model name for LLM agent")
    parser.add_argument("--output",    default=None,
                        help="Optional path to save metrics JSON")
    args = parser.parse_args()

    metrics = run_eval(
        split=args.split,
        agent=args.agent,
        verbose=args.verbose,
        partial=args.partial,
        llm_url=args.llm_url,
        llm_model=args.llm_model,
    )

    if args.output:
        # Exclude per_task detail for cleaner JSON file
        out = {k: v for k, v in metrics.items() if k != "per_task"}
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Metrics saved to {args.output}")


if __name__ == "__main__":
    main()