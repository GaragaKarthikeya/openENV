# Linux SRE Gym Implementation Plan

This document turns the hackathon problem statement into a concrete build plan for the current `linux_sre_gym` repository.

## 1. Goal

Replace the starter echo environment with a real OpenEnv-compatible Linux SRE simulator where an agent:

- explores a simulated Linux system through bash-like commands,
- diagnoses stateful incidents,
- applies safe configuration changes,
- receives dense rewards during the episode,
- is graded deterministically across 3 tasks:
  - Task 1: Triage
  - Task 2: Optimization
  - Task 3: Security Hardening

## 2. Current Repo Status

The repo already has a usable OpenEnv scaffold:

- `openenv.yaml`
- `server/app.py`
- `server/linux_sre_gym_environment.py`
- `client.py`
- `models.py`
- `Dockerfile`
- `README.md`

What is still missing for the hackathon:

- real Linux SRE simulation logic,
- typed action/observation/reward models aligned to the simulation,
- 3 deterministic tasks with graders,
- dense reward shaping,
- reproducible `inference.py` in repo root,
- README rewritten for the real environment,
- validation and deployment workflow for HF Spaces.

## 3. Target Architecture

Build the project around 5 layers:

1. Environment API layer
- OpenEnv `reset()`, `step()`, `state()`
- typed Pydantic models

2. Linux simulator layer
- mock `/proc`, `/sys`, `sysctl`, process table, network config
- command interpreter for a safe subset of bash commands

3. Task layer
- task definitions
- initial machine state per task
- completion rules

4. Reward and grading layer
- partial credit for discovery and correct remediation
- penalties for destructive, repeated, or irrelevant actions

5. Evaluation layer
- `inference.py`
- reproducible runs
- logging in the required `[START] / [STEP] / [END]` format

## 4. Recommended File Layout

Refactor the repo toward this structure:

```text
linux_sre_gym/
├── __init__.py
├── client.py
├── models.py
├── openenv.yaml
├── README.md
├── inference.py
├── implementation.md
├── Dockerfile
├── pyproject.toml
├── server/
│   ├── __init__.py
│   ├── app.py
│   └── linux_sre_gym_environment.py
├── simulator/
│   ├── __init__.py
│   ├── command_router.py
│   ├── kernel_state.py
│   ├── procfs.py
│   ├── sysctl.py
│   ├── process_table.py
│   └── networking.py
├── tasks/
│   ├── __init__.py
│   ├── base.py
│   ├── triage.py
│   ├── optimization.py
│   └── security.py
└── graders/
    ├── __init__.py
    ├── common.py
    ├── triage_grader.py
    ├── optimization_grader.py
    └── security_grader.py
```

You can keep everything under `server/` if you want fewer files, but splitting simulator/tasks/graders will make the environment easier to debug and explain.

## 5. Step-by-Step Implementation

### Phase 1: Fix the data model and environment contract

Objective: define the exact API the agent will use.

Update `models.py` to include:

- `LinuxSreGymAction`
  - `command: str`
- `LinuxSreGymObservation`
  - `stdout: str`
  - `stderr: str`
  - `exit_code: int`
  - `current_task: str`
  - `step_count: int`
  - `last_reward_reason: str`
  - `available_hint: str | None`
- `LinuxSreGymRewardBreakdown`
  - `discovery_reward: float`
  - `progress_reward: float`
  - `safety_penalty: float`
  - `repeat_penalty: float`
  - `completion_bonus: float`
- `LinuxSreGymState`
  - `task_id: str`
  - `filesystem: dict`
  - `processes: dict`
  - `sysctl: dict`
  - `network: dict`
  - `command_history: list[str]`
  - `reward_history: list[float]`
  - `is_resolved: bool`
  - `terminal_locked: bool`

Notes:

- Keep observation public and concise.
- Keep full hidden machine state in `state()`.
- Use explicit model names that match imports in `__init__.py` and `client.py`.

Definition of done:

- `client.py` can serialize `command`.
- `client.py` can parse the new observation fields.
- `state()` returns a typed Linux SRE state model, not only the generic episode state.

### Phase 2: Build the mock Linux machine

Objective: simulate enough Linux behavior to feel real without running a real shell.

Create an internal kernel state object with:

- process table:
  - PID
  - name
  - cpu percent
  - memory MB
  - status
  - killable bool
- virtual files:
  - `/proc/meminfo`
  - `/proc/loadavg`
  - `/proc/swaps`
  - `/proc/sys/vm/swappiness`
  - `/sys/module/zswap/parameters/enabled`
  - `/proc/sys/net/ipv4/conf/all/rp_filter`
  - `/proc/sys/net/ipv4/conf/default/rp_filter`
- runtime flags:
  - thrashing
  - spoofing_protection_enabled
  - runaway_process_present
  - service_health_score

Recommended command subset:

- `ls`
- `cat`
- `ps`
- `top` or a simplified `top -bn1`
- `grep`
- `sysctl`
- `echo VALUE > /proc/...` or `/sys/...`
- `kill`
- `pkill`
- `free -m`
- `vmstat`
- `dmesg`

Do not execute a real shell. Parse commands and dispatch them to Python handlers.

Definition of done:

- the simulator can return deterministic text output for all supported commands,
- unsupported commands return a realistic `stderr` and non-zero `exit_code`,
- writes modify simulator state and persist across steps in the same episode.

### Phase 3: Implement task 1, Task 2, Task 3

Objective: create three escalating, real-world SRE incidents.

#### Task 1: Triage

Initial state:

- one runaway process uses extreme CPU and memory,
- system load is high,
- memory pressure is visible in `/proc/meminfo`,
- `ps`, `top`, and `free -m` reveal the problem.

Expected successful actions:

- inspect process/memory state,
- identify offending PID or process name,
- terminate or reduce impact safely.

Completion criteria:

- runaway process no longer active,
- system load falls below threshold,
- memory headroom improves,
- no critical system process was killed.

#### Task 2: Optimization

Initial state:

- no runaway process,
- disk thrashing caused by bad paging config,
- `vm.swappiness` too high,
- `zswap` disabled,
- `vmstat`, `/proc/swaps`, `/proc/meminfo`, and `sysctl vm.swappiness` expose the issue.

Expected successful actions:

- inspect memory/paging state,
- lower swappiness to target value,
- enable zswap,
- verify changes.

Completion criteria:

- swappiness equals target,
- zswap enabled,
- thrashing flag cleared,
- agent used at least one diagnostic command before mutating state.

#### Task 3: Security Hardening

Initial state:

- networking stack accepts spoofed packets,
- reverse path filtering disabled,
- optional extra hardening flags start insecure.

Expected successful actions:

- inspect relevant sysctl networking paths,
- set `net.ipv4.conf.all.rp_filter=1`,
- set `net.ipv4.conf.default.rp_filter=1`,
- optionally verify via `sysctl -a | grep rp_filter`.

Completion criteria:

- both required sysctls are enabled,
- spoofing risk flag cleared,
- no unrelated dangerous network settings were disabled.

Definition of done:

- `reset()` can choose a task deterministically or via config,
- each task initializes a distinct machine state,
- each task has deterministic completion logic and grader output in `[0.0, 1.0]`.

### Phase 4: Design reward shaping

Objective: reward good SRE behavior over the full trajectory, not only final success.

Recommended reward components per step:

- `+0.05` for first-time useful discovery commands:
  - `cat /proc/meminfo`
  - `ps`
  - `top -bn1`
  - `vmstat`
  - `sysctl ...`
- `+0.10 to +0.25` for meaningful progress:
  - identifying correct PID,
  - changing a relevant sysctl,
  - enabling zswap,
  - fixing `rp_filter`
- `+0.40 to +0.60` completion bonus when the task is fully resolved
- `-0.02` repeated command penalty for useless repetition
- `-0.10 to -0.30` for destructive actions:
  - killing protected processes,
  - writing invalid sysctl values,
  - unrelated dangerous commands

Important rules:

- cap total score into `[0.0, 1.0]`,
- keep per-step rewards deterministic,
- use the same logic for both runtime reward and final grader,
- avoid sparse binary-only scoring.

Definition of done:

- each step returns a meaningful reward,
- random command spam scores worse than guided diagnosis,
- final episode score is normalized to `[0, 1]`.

### Phase 5: Implement the command interpreter

Objective: make the environment feel like terminal work.

Implementation strategy:

1. tokenize the command string,
2. match command family,
3. call a Python handler,
4. update machine state,
5. compute reward breakdown,
6. emit observation text.

Keep parsing intentionally narrow:

- support only patterns used in tasks,
- reject unsupported pipes/redirections unless explicitly implemented,
- add a few realistic aliases only if needed.

Suggested supported patterns:

- `cat <path>`
- `ps aux`
- `top -bn1`
- `free -m`
- `vmstat`
- `sysctl key`
- `sysctl -w key=value`
- `echo <value> > <path>`
- `kill <pid>`
- `pkill <name>`

Definition of done:

- the agent can solve all 3 tasks using only the supported command set,
- terminal output is stable and human-readable,
- invalid commands return realistic failure messages.

### Phase 6: Rewrite `server/linux_sre_gym_environment.py`

Objective: make the server environment the orchestrator over simulator + graders.

Responsibilities:

- create initial task state in `reset()`,
- route actions into simulator in `step()`,
- track step count and episode budget,
- compute reward and `done`,
- expose typed state via `state()`.

Recommended episode settings:

- `max_steps = 12` or `15`,
- done when:
  - task resolved,
  - max steps exceeded,
  - critical destructive action locks the task into failure.

Definition of done:

- `reset()` returns task-specific initial observation,
- `step()` is deterministic for the same state and action,
- `state()` includes everything needed for debugging and grading.

### Phase 7: Update `client.py` and `__init__.py`

Objective: keep the SDK usable for the baseline and external evaluators.

Update:

- `client.py` payload serialization to use `command`,
- observation parsing to use terminal-style outputs,
- exports in `__init__.py` to the final model names.

Definition of done:

- a local script can call `reset()`, `step()`, and `state()` without schema mismatch,
- package imports stay simple:
  - `from linux_sre_gym import LinuxSreGymAction, LinuxSreGymEnv`

### Phase 8: Create `inference.py` in repo root

Objective: satisfy the hackathon requirement exactly.

Requirements:

- file must be named `inference.py`,
- use the OpenAI client,
- read:
  - `API_BASE_URL`
  - `MODEL_NAME`
  - `HF_TOKEN`
  - optional local image variable if using Docker
- emit strict stdout logs:
  - `[START]`
  - `[STEP]`
  - `[END]`

Implementation plan:

1. run all 3 tasks sequentially,
2. for each task:
   - reset environment with that task id,
   - feed observation + history to model,
   - receive one command string,
   - call `env.step()`,
   - log immediately,
3. compute normalized score in `[0, 1]`,
4. always emit `[END]` even on exception.

Recommendation:

- add deterministic prompting,
- use `temperature=0` or low temperature for reproducibility,
- cap `MAX_STEPS` to stay far below the 20 minute limit.

Definition of done:

- `python inference.py` runs end-to-end,
- all logs follow the exact required format,
- the score is reproducible for a fixed model and task set.

### Phase 9: Rewrite `README.md`

Objective: make the project submission-ready.

README must include:

- environment motivation,
- why Linux SRE is a real-world agent benchmark,
- action space,
- observation space,
- state semantics,
- task descriptions and difficulty progression,
- reward shaping overview,
- setup instructions,
- local run instructions,
- Docker build/run instructions,
- HF Spaces deployment instructions,
- inference usage,
- expected baseline scores,
- limitations and future extensions.

Also explain clearly:

- this is a pure-Python simulation, not a real privileged shell,
- safety is preserved because commands are interpreted, not executed on the host.

Definition of done:

- README matches the final environment behavior,
- a reviewer can reproduce setup without reading source code first.

### Phase 10: Update `openenv.yaml` and Docker

Objective: pass validation and deploy cleanly.

`openenv.yaml` should include accurate metadata:

- name
- runtime/app entry
- port
- optional env vars for web interface and default task config

Docker plan:

- install project dependencies,
- expose port `8000`,
- run the FastAPI server,
- keep image small and deterministic.

Definition of done:

- `docker build .` succeeds,
- `docker run` starts the server cleanly,
- `/reset` responds with HTTP 200,
- `openenv validate` passes.

### Phase 11: Add local tests

Objective: catch regressions before submission.

Add tests for:

- task reset state,
- supported commands,
- invalid command behavior,
- grader score ranges,
- reward normalization,
- completion conditions for all 3 tasks.

Suggested minimal test cases:

- Task 1 solved by `ps aux` + `kill <pid>`
- Task 2 solved by checking `sysctl` + enabling `zswap`
- Task 3 solved by setting both `rp_filter` values
- repeated irrelevant commands accumulate penalties
- destructive wrong action reduces score

Definition of done:

- tests pass locally,
- graders never return values outside `[0.0, 1.0]`.

## 6. Detailed Grader Design

Each grader should return a deterministic float between `0.0` and `1.0`.

Suggested formula:

```text
score = clamp(
    discovery_score
    + remediation_score
    + verification_score
    - safety_penalty
    - repetition_penalty,
    0.0,
    1.0,
)
```

Suggested per-task rubric:

### Triage grader

- `0.20` inspected process or memory state
- `0.40` removed runaway process safely
- `0.20` system load/memory improved
- `0.20` verified outcome

### Optimization grader

- `0.20` inspected paging state
- `0.30` corrected swappiness
- `0.30` enabled zswap
- `0.20` verified reduced thrashing

### Security grader

- `0.20` inspected network sysctls
- `0.30` set `all.rp_filter`
- `0.30` set `default.rp_filter`
- `0.20` verified hardening

Penalty examples:

- `-0.10` repeated same command too often
- `-0.30` invalid dangerous write
- `-0.50` kill protected process in Task 1

## 7. Concrete Build Order

Use this order to avoid rework:

1. Finalize typed models in `models.py`
2. Update `client.py` and `__init__.py` to match those models
3. Implement simulator state object
4. Implement command handlers for read-only commands
5. Implement state-mutating commands
6. Implement task initialization for all 3 tasks
7. Implement graders and reward shaping
8. Replace the echo logic in `server/linux_sre_gym_environment.py`
9. Add `inference.py`
10. Rewrite `README.md`
11. Validate with Docker and `openenv validate`
12. Deploy to HF Spaces

## 8. Suggested Milestones

### Milestone A: Interactive prototype

You should be able to:

- reset Task 1,
- run `ps aux`,
- run `cat /proc/meminfo`,
- kill the runaway process,
- receive non-zero reward.

### Milestone B: Full task coverage

You should be able to solve all 3 tasks manually through the client.

### Milestone C: Baseline automation

`inference.py` should complete all tasks and print valid logs.

### Milestone D: Submission-ready

All of these should work:

- `docker build`
- `docker run`
- `openenv validate`
- local `inference.py`
- HF Space `/reset`

## 9. Validation Checklist

Before submission, verify:

- `inference.py` exists in repo root
- all LLM calls use `OpenAI` client
- env vars are read exactly as required
- three tasks are available and deterministic
- grader outputs stay in `[0, 1]`
- reward shaping is not binary-only
- Docker image builds successfully
- HF Space responds to `POST /reset`
- `openenv validate` passes
- README contains setup, tasks, action/observation spaces, and baseline scores

## 10. Risks to Avoid

- Do not call the real host shell for agent commands.
- Do not make the grader depend on nondeterministic text matching.
- Do not hide task success behind a single exact command string.
- Do not use sparse reward only at episode end.
- Do not forget strict logging format in `inference.py`.
- Do not leave the starter echo README in place.

## 11. Recommended Success Criteria

The project is submission-ready when:

- the environment clearly models real SRE diagnosis and remediation,
- each task is solvable through realistic command sequences,
- reward shaping encourages exploration before mutation,
- graders are deterministic and meaningful,
- the repo passes OpenEnv validation and Docker build,
- the HF Space responds correctly,
- the baseline script produces stable scores.

## 12. Optional Enhancements

If time remains, add:

- task randomization within safe deterministic templates,
- multiple valid remediation paths,
- richer `dmesg` and `journalctl` simulation,
- hidden distractor processes,
- command aliases,
- configurable difficulty levels,
- a benchmark mode that runs all tasks in sequence.

## 13. First Edits I Would Make in This Repo

If you want to start immediately, the first files to change are:

1. `models.py`
2. `client.py`
3. `server/linux_sre_gym_environment.py`
4. `README.md`
5. add `inference.py`

After that, create simulator/task/grader modules and wire them into the server.
