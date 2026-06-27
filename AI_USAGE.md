# AI_USAGE.md

## what i asked AI tools

i used Claude (claude.ai) throughout this assignment.

i started by sharing the assignment PDF and asking for a time breakdown, how long each part would take given 21 hours. then i asked for the full project design before writing any code, just to get the architecture clear in my head first.

after that i asked Claude to generate the code file by file: `state.py`, `actions.py`, `rubric.py`, `dataset.py`, `env.py`, `baseline.py`, `run_eval.py`, and the three test files.

i also asked it to rewrite the README and this file in my own writing style after sharing a sample of how i normally write.

---

## what code i accepted from AI tools

most of the initial code came from Claude. specifically:

* the full `state.py` with `AppState`, `SCREEN_ELEMENTS`, `NAVIGATION`, and `SCREEN_OBSERVATIONS`
* `actions.py` , `execute_action()`, `execute_action_list()`, `parse_agent_output()`
* `rubric.py` , all five reward functions and `compute_reward()`
* `dataset.py` , all 30 tasks and the `build_dataset()` / `get_task()` functions
* `env.py` , `MobileUIEnv`, `load_environment()`, and the `_EnvShim` fallback
* `baseline.py` , `heuristic_agent()`, `dummy_agent()`, `random_agent()`
* `run_eval.py` , the full eval loop with argparse and failure analysis
* the three test files as a starting point

---

## what i modified myself

a few things broke or were missing when i actually ran the code locally:

**`focus_mode_toggle` and `notifications_toggle` weren't doing anything** , tapping them fell through to the no-op return in `execute_action()` without actually flipping the boolean in state. two tests were failing because of this (`test_success_reward_focus_mode` and `test_success_reward_notifications_disabled`). i added the toggle handlers myself inside the tap branch.

**`baseline.py` was missing from my local directory** , the file existed in Claude's session but didn't make it into my project when i set it up. i had to manually create it.

**`__init__.py` was missing** , package imports were broken without it. created it manually.

**`environment.py` vs `env.py`** , my local file was named `environment.py` but the imports in `env.py` and tests expected `env`. had to align the naming.

**the test files looked too structured** , the original generated test files had numbered section comments, long docstrings, and a very formatted style that didn't look like something i'd write. i asked Claude to rewrite them simpler and then replaced the files. same for the README, rewrote it after sharing a sample of my writing.

**`notes_created` bug in baseline** , the original heuristic agent for multi-note tasks was navigating to the notes screen once per note instead of once total. Claude caught and fixed this when i pointed it out.

---

## what i learned

**reward conditioning matters more than i expected.** making `efficiency_reward` conditional on goal completion sounds obvious but i didn't think about it until Claude flagged that an unconditional version would reward the agent for finishing fast even when it failed the task. that's a subtle bug that would have been hard to catch just from reading the reward formula.

**sparse vs dense is about frequency, not just timing.** i thought sparse meant "fires at episode end" and dense meant "fires every step." but `invalid_action_penalty` fires every step it's triggered, which is sparse in practice if the agent rarely makes invalid moves. the better mental model is how often does this actually fire in a typical rollout.

**heuristic baselines expose reward bugs that random agents never find.** the random agent stress tests crash resistance. the heuristic agent runs the exact right sequence and catches cases where the reward function disagrees with the goal checker. running both told me different things.

**the verifiers shim pattern is underrated.** building `_EnvShim` as a fallback meant i could develop and test everything without the `verifiers` package installed while keeping `load_environment()` honest for when it is available. worth doing this for any optional dependency.

**environment design is mostly about what you don't allow.** most of the interesting decisions in this project were about what makes an action invalid, what ends an episode, and what the reward function ignores. the actual execution logic is straightforward. the constraints are where the design lives.
