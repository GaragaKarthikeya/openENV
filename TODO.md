# Linux SRE Gym - Team Task Breakdown

This document divides the implementation roadmap into three parallel tracks to minimize merge conflicts. Placeholder files and directories have already been generated in the codebase to help you get started immediately.

## 🧑‍💻 Person 1: The Core Architect (API, Server & Evaluation)
**Goal:** Define the exact contract between the AI agent and the environment, wire the pieces together, and build the final evaluation script.

**Your Files:**
*   `linux_sre_gym/models.py`
*   `linux_sre_gym/client.py`
*   `linux_sre_gym/server/linux_sre_gym_environment.py`
*   `inference.py` (in root)
*   `README.md` & `openenv.yaml`

**Your Tasks:**
1.  **Phase 1 & 7:** Update `models.py` with the exact schemas listed in the implementation plan (Action, Observation, State, etc.). Then, update `client.py` and `__init__.py` to use them. *Do this first so Person 2 and 3 know what data structures to expect.*
2.  **Phase 6:** Rewrite the `step()` and `reset()` methods in `linux_sre_gym_environment.py` to route commands to the Simulator (Person 2's code) and fetch scores from the Graders (Person 3's code).
3.  **Phase 8:** Write `inference.py` to use the OpenAI SDK and log exactly `[START]`, `[STEP]`, and `[END]` as strictly required by the hackathon.
4.  **Phases 9 & 10:** Update the `README.md` and `openenv.yaml` to ensure the project deploys properly to HF Spaces.

---

## 🐧 Person 2: The Kernel Hacker (Mock OS Simulator)
**Goal:** Build the fake Linux environment that reacts to bash commands without running a real shell.

**Your Files:**
*   `linux_sre_gym/simulator/kernel_state.py`
*   `linux_sre_gym/simulator/command_router.py`

**Your Tasks:**
1.  **Phase 2:** Build a Python class (`KernelState`) that holds a fake process table (list of dicts), a fake filesystem (dict mapping paths to string contents), and fake `sysctl` flags.
2.  **Phase 5:** Build `command_router.py`. This script should take a string like `cat /proc/meminfo` or `kill 1234`. It must:
    *   Parse the string safely.
    *   Look up the corresponding fake file/process in `KernelState`.
    *   Return a string `stdout`, a `stderr` (if it failed), and an `exit_code`.
    *   Support commands: `ls`, `cat`, `ps`, `top`, `grep`, `sysctl`, `echo >`, `kill`, `free`, `vmstat`.
3.  *Collaboration:* Expose a way for Person 3 to inject a specific `KernelState` when a task begins.

---

## 🕵️‍♂️ Person 3: The Scenarist (Tasks, Graders, & Tests)
**Goal:** Create the specific hackathon scenarios (the "levels") and the reward system that deterministically grades the AI's behavior.

**Your Files:**
*   `linux_sre_gym/tasks/triage.py` (and `.optimization.py`, `.security.py`)
*   `linux_sre_gym/graders/triage_grader.py` (and `.optimization_grader.py`, `.security_grader.py`)
*   *(New)* Unit Tests

**Your Tasks:**
1.  **Phase 3:** Write functions that generate the initial `KernelState` (defined by Person 2) for the 3 tasks:
    *   **Triage:** Inject a runaway process consuming CPU/RAM.
    *   **Optimization:** Set `vm.swappiness` too high and disable `zswap`.
    *   **Security:** Disable `rp_filter` network settings.
2.  **Phase 4:** Write the `Grader` classes. These should evaluate the current `KernelState` and the Agent's action history to return a deterministic score between `0.0` and `1.0`. Add penalties for destructive commands and rewards for good diagnostics.
3.  **Phase 11:** Write local tests to ensure your graders always return a float between 0 and 1, and that the tasks are completable using the commands Person 2 is building.

---
## 🤝 Merge Strategy
1. **Person 1** finishes `models.py` right away and pushes it.
2. **Person 2** and **Person 3** can now work completely independently in the `simulator/` and `tasks/` directories respectively.
3. Once Person 2 and 3 have a basic version of their modules, **Person 1** imports them into `server/linux_sre_gym_environment.py` and wires them up.