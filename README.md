# mobile-ui-env

an RL-style mobile UI agent environment built around Prime Intellect
Verifiers concepts.

a simulated mobile app where an AI agent completes tasks by producing
structured JSON actions and receives rewards based on task success,
action validity, efficiency, and safety.

------------------------------------------------------------------------

## setup

``` bash
git clone <repo>
cd mobile_ui_environment
pip install -e ".[dev]"
```

no external dependencies required. the `verifiers` package is optional,
if it's not installed, the environment falls back to a built-in shim
that has the same interface.

------------------------------------------------------------------------

## run evaluation

``` bash
# heuristic baseline on eval split (default)
python run_eval.py

# verbose per-task output
python run_eval.py --verbose

# dummy agent (always wrong) - verifies penalty logic works
python run_eval.py --agent dummy --verbose

# random agent - stress tests crash resistance
python run_eval.py --agent random --verbose

# run on train split
python run_eval.py --split train --verbose

# local LLM via ollama or any openai-compatible server
python run_eval.py --agent llm --llm-url http://localhost:11434/v1 --llm-model llama3

# save metrics to a json file
python run_eval.py --output metrics.json
```

## run tests

``` bash
pytest tests/ -v
pytest tests/ -v --tb=short
pytest tests/ --cov=mobile_ui_env
```

------------------------------------------------------------------------

## design Q&A

### 1. what is the state space?

the state is everything the environment needs to track within an
episode.

  ------------------------------------------------------------------------
  field                    type                    description
  ------------------------ ----------------------- -----------------------
  `current_screen`         `str`                   one of `home`, `notes`,
                                                   `settings`, `profile`

  `notes`                  `list[str]`             note titles saved so
                                                   far this episode

  `focus_mode`             `bool`                  whether focus mode is
                                                   on

  `notifications`          `bool`                  whether notifications
                                                   are on

  `username`               `str`                   read-only label :
                                                   `jane_doe`

  `email`                  `str`                   read-only label :
                                                   `jane@example.com`

  `app_version`            `str`                   read-only label :
                                                   `2.4.1`

  `note_input_buffer`      `str`                   text typed but not
                                                   saved yet

  `invalid_action_count`   `int`                   how many invalid
                                                   actions taken this
                                                   episode

  `safety_violation`       `bool`                  true if logout button
                                                   was tapped

  `steps_taken`            `int`                   total actions executed
                                                   so far

  `finished`               `bool`                  true if agent called
                                                   finish
  ------------------------------------------------------------------------

the agent sees a natural language observation string describing the
current screen and what's available on it. dynamic values like focus
mode status, username, notes count are embedded in the observation text.

------------------------------------------------------------------------

### 2. what is the action space?

four action types:

  ----------------------------------------------------------------------------------------------------------------
  action                  schema                                                           what it does
  ----------------------- ---------------------------------------------------------------- -----------------------
  `tap`                   `{"action": "tap", "target": "<element>"}`                       tap a UI element

  `type`                  `{"action": "type", "target": "note_input", "text": "<text>"}`   type into the note
                                                                                           input field

  `back`                  `{"action": "back"}`                                             navigate back to home

  `finish`                `{"action": "finish"}`                                           tell the environment
                                                                                           you're done
  ----------------------------------------------------------------------------------------------------------------

the agent outputs a list of these actions in a single turn. invalid
targets, missing keys, wrong screen none of these crash the environment,
they just get penalised.

------------------------------------------------------------------------

### 3. what is the episode termination condition?

an episode ends when any of these happens:

1.  **agent calls `finish`** : agent signals it's done.
2.  **`steps_taken >= max_steps`** : hard step budget is exhausted.
3.  **`safety_violation = True`** : agent tapped logout, episode ends
    immediately with penalty applied.

reward is always computed from the final state regardless of how the
episode ended.

------------------------------------------------------------------------

### 4. which rewards are sparse?

  -----------------------------------------------------------------------
  reward                  type                    why
  ----------------------- ----------------------- -----------------------
  `success_reward`        **sparse**              1.0 only if goal is met
                                                  at the end, 0
                                                  otherwise - no signal
                                                  mid-episode

  `safety_penalty`        **sparse**              −0.3 only if logout was
                                                  triggered, 0 otherwise
  -----------------------------------------------------------------------

sparse rewards are hard for RL agents because there's no gradient signal
until the agent accidentally completes the task. in a 4-screen
environment like this, an agent exploring randomly has maybe a
1-in-thousands chance of completing a composite task. that means
millions of rollouts before the reward signal has enough support to
shape any useful behaviour.

------------------------------------------------------------------------

### 5. which rewards are dense or shaped?

  ---------------------------------------------------------------------------
  reward                      type                    why
  --------------------------- ----------------------- -----------------------
  `format_reward`             **dense**               fraction of actions
                                                      with valid structure,
                                                      fires every step

  `efficiency_reward`         **shaped**              scales inversely with
                                                      steps used, but only
                                                      when goal is met

  `invalid_action_penalty`    **dense**               −0.1 per invalid
                                                      action, immediate
                                                      feedback

  `partial_progress_reward`   **shaped (optional)**   +0.3 for reaching the
                                                      right screen, helps
                                                      with credit assignment
  ---------------------------------------------------------------------------

shaped rewards help bridge the credit assignment gap by giving
intermediate signal before the terminal goal is reached.

`efficiency_reward` is conditioned on success intentionally, the agent
can't earn efficiency points by finishing fast without actually
completing the goal.

------------------------------------------------------------------------

### 6. how can reward hacking happen in this environment?

a few ways:

**screen farming** : if `partial_progress_reward` is on, an agent could
learn to navigate back and forth to the target screen repeatedly to farm
the +0.3 without ever completing the goal. mitigation: cap it at 0.5,
disable it for eval.

**format gaming** : an agent optimising only `format_reward` could learn
to output syntactically valid but useless actions like immediately
calling `finish`. the weight (0.1) keeps this from dominating but it's
still possible early in training.

**efficiency hacking** : if the goal checker used fuzzy matching instead
of exact match, an agent could complete a shorter wrong task and still
claim efficiency points. exact match on note titles prevents this.

**invalid action fishing** : an agent could figure out that some invalid
actions are "free" because the −0.1 penalty is less than what it earns
elsewhere. `clip(final, 0, 1)` prevents catastrophic negatives but
doesn't fully stop this.

the right fix is keeping `success_reward` weight dominant at 1.0, strict
goal matching, and separated train/eval tasks so the agent can't just
memorise answers.

------------------------------------------------------------------------

### 7. how would you scale this from a mock UI to a real android emulator?

the mock environment has four components that map directly to real
android equivalents:

  -----------------------------------------------------------------------
  mock                                real android equivalent
  ----------------------------------- -----------------------------------
  `SCREEN_ELEMENTS` dict              UI hierarchy / accessibility tree
                                      via `adb shell uiautomator dump`

  `get_observation()` string          screenshot + XML tree parsed into
                                      text, or fed to a vision model

  `execute_action()`                  `adb shell input tap x y` or
                                      `adb shell input text` with
                                      coordinate mapping

  `AppState`                          full emulator snapshot
  -----------------------------------------------------------------------

migration path: 1. swap `SCREEN_ELEMENTS` for a live accessibility tree
parser (uiautomator2 or Android ADK) 2. swap `execute_action()` for an
ADB command executor 3. swap `get_observation()` for a screenshot + XML
tree extractor 4. swap manual goal checkers for a VLM-based judge or
pixel/tree diffing

the key thing is the environment's interface : `reset()`, `step()`,
`compute_reward()` - is already decoupled from the implementation.
swapping the mock backend for a real emulator doesn't touch the reward
functions, dataset, or training loop at all.

------------------------------------------------------------------------

### 8. how would this work with Prime Intellect, Verifiers, or PRIME-RL?

`load_environment()` returns a `vf.SingleTurnEnv` when the verifiers
package is installed:

``` python
import verifiers as vf
from mobile_ui_env import load_environment

env = load_environment()

trainer = vf.GRPOTrainer(
    model=model,
    env=env,
    ...
)
trainer.train()
```

the reward functions are standalone callables with signature
`(state, task, actions) -> float`, which maps directly to
`vf.Rubric(funcs=[...], weights=[...])`.

with PRIME-RL or distributed training on Prime Intellect compute: - each
worker node runs independent rollouts via
`vf.SingleTurnEnv.generate()` - rewards are computed locally and
aggregated for GRPO/PPO updates - train/eval split ensures the model is
never evaluated on tasks it trained on - `reset()` is cheap and
stateless so the environment parallelises trivially

------------------------------------------------------------------------

### 9. what tests did you write?

**`tests/test_actions.py`** (21 tests) - valid tap changes screen -
invalid tap doesn't crash the environment - invalid taps accumulate
count correctly - missing action key and unknown action type handled -
full note creation flow updates state - saving without typing first is
penalised - two notes created in sequence - typing into a non-typeable
element is penalised - back navigation returns to home - finish sets
done flag - action list respects max_steps - action list stops at finish
action - parse_agent_output handles valid JSON, bad JSON, and non-list
JSON

**`tests/test_rewards.py`** (16 tests) - success_reward = 1.0 for note,
focus mode, notifications tasks - success_reward = 0.0 for wrong title
and incomplete task - logout triggers safety_penalty = −1.0 - no safety
penalty without logout - invalid actions accumulate penalty correctly -
no penalty for clean run - format_reward on all valid / partial / empty
action lists - efficiency_reward is 1.0 at min_steps, 0.0 at max_steps,
0.0 if goal not met - compute_reward clips to \[0, 1\] - clean success
run yields final \> 0.9 - safety penalty still applies even when goal is
met

**`tests/test_environment.py`** (14 tests) - reset() returns an
observation string - step() returns the right types - reward is bounded
\[0, 1\] - info dict has all expected keys - episode ends at max_steps -
episode ends on finish - step after done returns error in info - logout
sets safety_violation - logout reduces reward vs clean visit -
load_environment() returns datasets of correct size - dataset splits are
correctly assigned - shim run_episode works end to end - state fully
resets between episodes

------------------------------------------------------------------------

### 10. what tradeoffs did you make?

**single-turn vs multi-turn** : the environment takes the full action
list in one `step()` call. a real training loop would call `step()` once
per action so the agent can observe intermediate state. the interface
supports that - just pass single-action lists but the baseline and eval
script use single-turn for simplicity.

**simulated toggles** : `focus_mode_toggle` and `notifications_toggle`
flip on every tap regardless of current state. in a real app the agent
would need to read current state first. this simplification avoids the
agent needing to parse booleans from the observation.

**read-only goals** : for `read_username`, `read_email`, `read_version`,
the environment checks that the agent reached the right screen, not that
it actually extracted and returned the value. doing that properly would
need a generative output parser.

**text-only observation** : no screenshots. a real mobile agent would
need pixel or XML tree input. adding image support means integrating a
multimodal model and an emulator rendering pipeline.

**partial_progress_reward off by default** : reward hacking risk (screen
farming) outweighs the benefit for a 4-screen environment where sparse
reward is reachable within a reasonable number of rollouts.

**exact match on note titles** : case-insensitive with whitespace
stripping, but no fuzzy matching. intentional, fuzzy matching opens a
reward hacking surface.
