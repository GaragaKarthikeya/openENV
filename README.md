---
title: Linux SRE Gym
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - systems
---

# Linux SRE Gym

Linux SRE Gym is a pure-Python OpenEnv benchmark for autonomous server diagnostics and remediation. Instead of running commands on a real host, it simulates a Linux machine with mock `procfs`, `sysfs`, `sysctl`, and process-table state, then lets an agent interact through bash-like commands such as `ps aux`, `cat /proc/meminfo`, `sysctl -w ...`, and `kill <pid>`.

The goal is to evaluate whether an agent can behave like an SRE: inspect system state safely, form a diagnosis, and apply the right fix without damaging the machine.

## Why This Environment Exists

Single-turn code tasks are not enough for operations work. Real infrastructure debugging is sequential and stateful:

- the agent has to discover hidden state,
- the next command depends on prior terminal output,
- remediation must be safe,
- success depends on the final machine state, not only the last response.

Linux SRE Gym targets that gap with three deterministic incidents that escalate from triage to kernel tuning to network hardening.

## Tasks

The environment cycles deterministically through three tasks on consecutive `reset()` calls unless `LINUX_SRE_GYM_DEFAULT_TASK` is set.

### 1. Triage

Identify and mitigate a runaway process that is saturating CPU and memory.

Expected behaviors:

- inspect `ps aux`, `top -bn1`, `free -m`, `/proc/meminfo`
- identify the offender
- terminate the runaway process without killing protected system services

### 2. Security Hardening

Harden networking so spoofed packets are dropped.

Expected behaviors:

- inspect `rp_filter` values via `/proc/sys` or `sysctl`
- set `net.ipv4.conf.all.rp_filter=1`
- set `net.ipv4.conf.default.rp_filter=1`
- verify the hardening took effect

### 3. Optimization

Resolve disk thrashing caused by bad paging settings.

Expected behaviors:

- inspect `vmstat`, `/proc/swaps`, `/proc/meminfo`, `sysctl vm.swappiness`
- reduce `vm.swappiness`
- enable `zswap`
- verify the host is healthier after the changes

## Action Space

`LinuxSreGymAction`

- `command: str`

The agent sends exactly one bash-like command per step.

Supported command families in the built-in simulator:

- `ps aux`
- `top -bn1`
- `free -m`
- `vmstat`
- `cat <path>`
- `ls <path>`
- `grep <pattern> <path>`
- `sysctl <key>`
- `sysctl -w <key>=<value>`
- `sysctl -a | grep rp_filter`
- `echo <value> > <path>`
- `kill <pid>`
- `pkill <name>`

## Observation Space

`LinuxSreGymObservation`

- `stdout: str`
- `stderr: str`
- `exit_code: int`
- `current_task: str`
- `step_count: int`
- `last_reward_reason: str`
- `available_hint: str | None`
- `reward: float`
- `done: bool`
- `reward_breakdown: LinuxSreGymRewardBreakdown`
- `metadata: dict`

This mirrors a terminal interaction while still exposing structured reward details to the agent and evaluator.

## State Space

`LinuxSreGymState`

- `episode_id`
- `task_id`
- `task_description`
- `filesystem`
- `processes`
- `sysctl`
- `network`
- `command_history`
- `reward_history`
- `seen_diagnostics`
- `is_resolved`
- `terminal_locked`
- `completion_score`

`state()` is intended for debugging and grading, while the agent should primarily reason from observations.

## Reward Design

Reward is dense and rubric-driven instead of binary-only.

Positive signals:

- first-time diagnostic commands for the active task
- relevant remediation actions
- completion bonus when the incident is fully resolved

Negative signals:

- repeated low-value commands
- invalid writes
- destructive actions such as killing protected processes

Each task also has a deterministic normalized score in `[0.0, 1.0]` used for evaluation.

## Project Layout

```text
.
├── __init__.py
├── client.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── README.md
├── Dockerfile
├── inference.py
├── simulator/
├── tasks/
├── graders/
└── server/
    ├── app.py
    └── linux_sre_gym_environment.py
```

The server includes a built-in deterministic simulator and grader as the default backend.

## Local Setup

```bash
cd linux_sre_gym
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you want the full dev toolchain:

```bash
pip install -e ".[dev]"
```

For local validation and tests, install the dev extras so `pytest` is available in the same environment as `openenv`.

Run the test suite from the repo root:

```bash
pytest -q
```

## Run the Server Locally

```bash
cd linux_sre_gym
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Important endpoints:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /docs`
- `GET /health`
- `WS /ws`

## Python Client Example

```python
from linux_sre_gym import LinuxSreGymAction, LinuxSreGymEnv

with LinuxSreGymEnv(base_url="http://127.0.0.1:8000") as env:
    result = env.reset()
    print(result.observation.current_task)
    print(result.observation.stdout)

    result = env.step(LinuxSreGymAction(command="ps aux"))
    print(result.observation.stdout)
    print(result.reward, result.done)
```

## Docker

Build:

```bash
cd linux_sre_gym
docker build -t linux-sre-gym:latest .
```

Run:

```bash
docker run --rm -p 8000:8000 linux-sre-gym:latest
```

## OpenEnv Validation

Run the validator from the environment directory:

```bash
cd linux_sre_gym
openenv validate
```

The hackathon validation flow also expects:

- a working Docker build
- a deployed HF Space responding to `POST /reset`
- a root-level `inference.py`

## Baseline Inference

- [`inference.py`](./inference.py)

Required environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Optional:

- `LOCAL_IMAGE_NAME` to launch the env from a local Docker image
- `ENV_BASE_URL` to connect to an already-running server
- `LINUX_SRE_GYM_MAX_STEPS`
- `SUCCESS_SCORE_THRESHOLD`

Run it from the repo root. The script bootstraps the local package automatically, so it works both from a raw checkout and after `pip install -e .`:

```bash
python3 inference.py
```

The script emits only the hackathon-required structured logs:

- `[START]`
- `[STEP]`
- `[END]`

## Hugging Face Spaces Deployment

This repo is configured for Docker-based HF Spaces. From the environment directory:

```bash
cd linux_sre_gym
openenv push
```

Or build and push manually using your Space's Docker workflow.

## Baseline Scores

Measured locally against the final benchmark via Hugging Face Inference Providers on 8 April 2026 with `MAX_STEPS=12` and `SUCCESS_SCORE_THRESHOLD=0.85`. The table below includes only models that completed all 3 tasks.

| Model | Triage | Security | Optimization | Avg | Outcome |
|------|-------|----------|--------------|-----|---------|
| `google/gemma-4-31B-it` | 0.999 | 0.950 | 0.950 | 0.966 | Solves all 3 tasks |
| `Qwen/Qwen2.5-72B-Instruct` | 0.999 | 0.900 | 0.650 | 0.850 | Solves triage + security, fails optimization |
| `Qwen/Qwen3-235B-A22B-Instruct-2507` | 0.999 | 0.800 | 0.950 | 0.916 | Solves triage + optimization, fails security |

Takeaways:

- **Triage** is reliably solved by strong instruction models.
- **Security** is a medium task because the agent must both harden `rp_filter` and verify the fix afterward.
- **Optimization** is still challenging, but fairer now that sensible paging and zswap probes receive reward credit.

The most common failure modes are weak verification discipline on `security` and unnecessary or repetitive sysctl tuning before final verification on `optimization`.

## Safety Model

This environment does not execute host shell commands. All commands are parsed and interpreted inside Python against simulated machine state. That keeps the benchmark safe to run locally, inside CI, and on Hugging Face Spaces without privileged access.
