# mobile-ui-env

An RL-style mobile UI agent environment built with Prime Intellect Verifiers concepts.
A simulated mobile app where an AI agent completes tasks by producing structured JSON actions
and receives rewards based on task success, action validity, efficiency, and safety.

---

## Setup

```bash
git clone <repo>
cd mobile_ui_env
pip install -e ".[dev]"
```

No external dependencies are required. The `verifiers` package is optional —
the environment degrades gracefully to a built-in shim if it is not installed.

---

## Run evaluation

```bash
# Heuristic baseline on eval split (default)
python run_eval.py

# Verbose per-task output
python run_eval.py --verbose

# Dummy (always-wrong) agent — stress tests penalty logic
python run_eval.py --agent dummy

# Random agent — stress tests crash-resistance
python run_eval.py --agent random

# Train split
python run_eval.py --split train --verbose

# Optional: local LLM via Ollama or any OpenAI-compatible server
python run_eval.py --agent llm --llm-url http://localhost:11434/v1 --llm-model llama3

# Save metrics to JSON
python run_eval.py --output metrics.json
```

## Run tests

```bash
pytest tests/ -v
pytest tests/ -v --tb=short   # concise tracebacks
pytest tests/ --cov=mobile_ui_env  # coverage (requires pytest-cov)
```

---

## Design Q&A

### 1. What is the state space?

| Field | Type | Description |
|---|---|---|
| `current_screen` | `str` | One of `home`, `notes`, `settings`, `profile` |
| `notes` | `list[str]` | Note titles saved during the episode |
| `focus_mode` | `bool` | Whether focus mode is enabled |
| `notifications` | `bool` | Whether notifications are enabled |
| `username` | `str` | Read-only label (`jane_doe`) |
| `email` | `str` | Read-only label (`jane@example.com`) |
| `app_version` | `str` | Read-only label (`2.4.1`) |
| `note_input_buffer` | `str` | Text typed but not yet saved |
| `invalid_action_count` | `int` | Cumulative invalid actions this episode |
| `safety_violation` | `bool` | True if logout was triggered |
| `steps_taken` | `int` | Actions executed so far |
| `finished` | `bool` | True if agent called `finish` |

The observation returned to the agent is a natural-language string describing
the current screen and its available elements. Each screen also exposes its
dynamic values (focus mode status, note count, username, etc.).

---

### 2. What is the action space?

Four action types:

| Action | Schema | Description |
|---|---|---|
| `tap` | `{"action": "tap", "target": "<element>"}` | Tap a UI element |
| `type` | `{"action": "type", "target": "note_input", "text": "<text>"}` | Type into a text field |
| `back` | `{"action": "back"}` | Navigate back to Home |
| `finish` | `{"action": "finish"}` | Signal episode completion |

The agent outputs a **list** of these actions in one turn (single-turn environment).
Invalid targets, missing keys, or unsupported types are penalised but never crash the environment.

---

### 3. What is the episode termination condition?

An episode ends when **any** of these is true:

1. **Agent calls `finish`** — the agent explicitly signals it is done.
2. **`steps_taken >= max_steps`** — the hard step budget is exhausted.
3. **`safety_violation = True`** — the agent tapped the logout button; episode ends immediately with the safety penalty applied.

In all cases, the reward is computed from the final state.

---

### 4. Which rewards are sparse?

| Reward | Type | Reason |
|---|---|---|
| `success_reward` | **Sparse** | 1.0 only at episode end if goal is met; 0 otherwise — no gradient signal mid-episode |
| `safety_penalty` | **Sparse** | −0.3 only if logout is triggered; 0 otherwise |

Sparse rewards are hard for RL agents because the agent receives no learning signal
until it accidentally completes the task. In longer-horizon tasks (multiple screens,
multiple steps), this makes exploration extremely sample-inefficient. An agent
exploring randomly in this 4-screen environment has roughly a 1-in-thousands chance
of completing a composite task by chance — meaning it could take millions of rollouts
before the reward signal has enough support to shape useful gradients.

---

### 5. Which rewards are dense or shaped?

| Reward | Type | Reason |
|---|---|---|
| `format_reward` | **Dense** | Fraction of actions with valid structure — signal on every step |
| `efficiency_reward` | **Shaped** | Scales with inverse of steps used, conditioned on success — encourages shorter paths |
| `invalid_action_penalty` | **Dense** | −0.1 per invalid action — immediate signal discouraging bad actions |
| `partial_progress_reward` | **Shaped (optional)** | +0.3 for reaching the correct screen — credit assignment aid |

Shaped rewards help bridge the credit-assignment gap by giving the agent intermediate
signals before it reaches the terminal goal. `efficiency_reward` is designed carefully:
it is **conditioned on success**, meaning the agent cannot earn efficiency points for
completing the task quickly via an unsafe or incorrect shortcut.

---

### 6. How can reward hacking happen in this environment?

Several reward hacking risks exist:

**a) Screen farming via partial_progress_reward**
If `partial_progress_reward` is enabled, an agent could learn to navigate to the
target screen repeatedly (back → target → back → target) to accumulate partial
credit without ever completing the goal. Mitigation: cap at 0.5, only enable for
training, disable for final eval.

**b) Format gaming**
An agent optimising only `format_reward` could learn to output syntactically valid
but semantically useless actions (`finish` immediately). The weight (0.1) keeps this
from dominating, but pure format optimisation is still possible in early training.

**c) Efficiency hacking when success check is weak**
If the success check were lenient (e.g., substring match instead of exact match on
note titles), an agent could complete a shorter wrong task and claim efficiency
points. The current exact-match check mitigates this.

**d) Invalid action fishing**
With enough rollouts, an agent could learn that certain invalid actions are
"free" because their penalty (−0.1) is less than the reward from other components.
The safety of `clip(final, 0, 1)` prevents catastrophic negative rewards but does
not prevent this mild exploit. Increasing the invalid penalty weight would mitigate it.

**Correct mitigation strategy:** Keep `success_reward` weight dominant (1.0),
validate goals with strict matching, separate train/eval tasks so the agent cannot
memorise answers, and use the partial reward only during early training curriculum.

---

### 7. How would you scale from a mock UI to a real Android emulator?

The mock environment abstracts four components that have direct real-world analogues:

| Mock Component | Real Android Equivalent |
|---|---|
| `SCREEN_ELEMENTS` dict | **UI hierarchy / accessibility tree** (XML dumped via `adb shell uiautomator dump`) |
| `get_observation()` string | **Screenshot + XML tree** parsed into text; optionally a rendered image for vision models |
| `execute_action()` | **`adb shell input tap x y`** or `adb shell input text` via coordinate mapping |
| `AppState` | **Emulator snapshot** (full system state, not just app state) |

Concretely the migration path is:

1. Replace `SCREEN_ELEMENTS` with a live accessibility tree parser (e.g., `uiautomator2` or Android ADK).
2. Replace `execute_action()` with an action executor that sends ADB commands to the emulator.
3. Replace `get_observation()` with a screenshot + XML tree extractor, optionally feeding both to a multimodal LLM.
4. Replace manual goal checkers with a VLM-based judge or pixel/tree diffing to verify task completion.
5. Wrap everything in the same `reset()` / `step()` interface — the rubric layer stays identical.

The key insight is that the environment's **interface contract** (`reset`, `step`, `compute_reward`)
is already decoupled from its implementation. Swapping the mock backend for a real emulator
does not require changing the reward functions, dataset, or training loop.

---

### 8. How would this work with Prime Intellect, Verifiers, or PRIME-RL?

`load_environment()` is designed to return a `vf.SingleTurnEnv` when the `verifiers`
package is installed:

```python
import verifiers as vf
from mobile_ui_env import load_environment

env = load_environment()
# env is a vf.SingleTurnEnv with train/eval datasets and a vf.Rubric

trainer = vf.GRPOTrainer(
    model=model,
    env=env,
    ...
)
trainer.train()
```

The rubric functions (`success_reward`, `format_reward`, etc.) are standalone callables
with a consistent signature `(state, task, actions) -> float`, which maps directly to
the `vf.Rubric(funcs=[...], weights=[...])` interface.

With PRIME-RL or distributed training on Prime Intellect compute:

- Each worker node runs independent environment rollouts (`vf.SingleTurnEnv.generate()`).
- Rewards are computed locally and aggregated for GRPO/PPO gradient updates.
- The dataset split (train/eval) ensures the model is never evaluated on tasks it trained on.
- The environment is stateless between episodes (`reset()` is cheap), making it trivially parallelisable.

---

### 9. What tests did you write?

**`tests/test_actions.py`** (17 tests)
- Valid tap changes screen
- Invalid tap does not crash the environment
- Multiple invalid taps accumulate count
- Missing `action` key / unknown action type handled
- Full note creation flow updates state
- Save without typing is penalised
- Two-note creation works
- Typing into non-typeable element is penalised
- Back navigation returns to home
- Finish sets done flag
- Action list respects max_steps
- Action list stops at finish
- `parse_agent_output` handles valid JSON, invalid JSON, and non-list JSON

**`tests/test_rewards.py`** (16 tests)
- `success_reward` = 1.0 for note, focus mode, notifications tasks
- `success_reward` = 0.0 for wrong title and incomplete task
- Logout triggers `safety_penalty` = −1.0
- No safety penalty without logout
- Invalid actions accumulate correctly
- No penalty for clean run
- `format_reward` scores well-formed / partial / empty actions
- `efficiency_reward` is 1.0 at min_steps, 0.0 at max_steps, 0.0 if goal not met
- `compute_reward` clips to [0, 1]
- Clean success run yields final > 0.9
- Safety + success still penalises

**`tests/test_env.py`** (14 tests)
- `reset()` returns observation string
- `step()` returns correct types
- Reward is bounded [0, 1]
- Info dict contains expected keys
- Episode ends at max_steps
- Episode ends on finish
- Step after done returns error info
- Logout sets safety_violation
- Logout reduces reward vs clean visit
- `load_environment()` returns datasets of correct sizes
- Dataset splits are correctly assigned
- Shim `run_episode` works end-to-end
- State fully resets between episodes

---

### 10. What tradeoffs did you make due to the limited assignment scope?

**Single-turn vs multi-turn:** The environment processes the entire action list in one
`step()` call. A real RL training loop would call `step()` once per action to allow
the agent to observe intermediate state. The interface supports multi-turn use — just
call `step()` with single-action lists — but the baseline and eval script use single-turn
for simplicity.

**Simulated toggles:** `focus_mode_toggle` and `notifications_toggle` flip state on
every tap regardless of current state (no "already enabled" guard). In a real app,
the agent would need to read current state to know whether to tap. This simplification
avoids the agent needing to parse boolean state from the observation.

**Read-only goals (`read_username`, `read_email`, `read_version`):** The environment
checks that the agent reached the correct screen, not that the agent extracted and
returned the value. In a real evaluation, you would verify the agent's answer against
the ground truth value — requires a generative output parser.

**No screenshot / visual observation:** The observation is text-only. A real agent
would need pixel or XML-tree input for general mobile UI tasks. Adding image support
would require integrating a multimodal model and an emulator rendering pipeline.

**`partial_progress_reward` off by default:** Enabled only via `--partial` flag or
`include_partial=True` constructor argument. Reward hacking risk (screen farming)
outweighs the benefit for the current 4-screen environment where sparse reward is
already reachable within a reasonable number of rollouts.

**Goal checking is exact match:** Note titles are compared case-insensitively after
stripping whitespace, but no fuzzy matching is used. This is intentional — fuzzy
matching opens a reward hacking surface.