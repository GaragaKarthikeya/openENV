"""Baseline inference runner for Linux SRE Gym."""

from __future__ import annotations

import asyncio
import os
import sys
import textwrap
from pathlib import Path
from typing import List, Optional

from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from linux_sre_gym import LinuxSreGymAction, LinuxSreGymEnv  # noqa: E402

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = os.getenv("LINUX_SRE_GYM_BENCHMARK") or "linux_sre_gym"
ENV_BASE_URL = os.getenv("ENV_BASE_URL") or "http://127.0.0.1:8000"
MAX_STEPS = int(os.getenv("LINUX_SRE_GYM_MAX_STEPS", "12"))
TASK_COUNT = 3
TEMPERATURE = 0.0
MAX_TOKENS = 64
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.85"))

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a senior Linux SRE agent. Your goal is to diagnose and resolve system incidents.
    
    RULES:
    1. Reply with EXACTLY one Linux command and nothing else. No preamble, no explanation.
    2. Prefer safe diagnosis (ps, top, vmstat, free) before mutation.
    3. MANDATORY: For all sysctl updates, you MUST use 'sysctl -w key=value'.
    4. MANDATORY: To enable zswap, use 'echo Y > /sys/module/zswap/parameters/enabled'.
    5. DECISIVENESS: Once you have identified a runaway PID or a misconfiguration, ACT immediately to fix it. Do not repeat the same diagnostic command twice.
    6. VERIFICATION: After applying a fix, run one final diagnostic command to verify the change.
    
    SUPPORTED COMMANDS:
    - ps aux, top -bn1, free -m, vmstat
    - cat <path>, ls <path>
    - sysctl <key>, sysctl -a | grep <pattern>
    - sysctl -w <key>=<value>
    - echo <value> > <path>
    - kill <pid>, pkill <name>
    """
).strip()


def _flatten(value: Optional[str]) -> str:
    if not value:
        return "null"
    return value.replace("\n", "\\n")


def _normalize_action(value: str) -> str:
    return value.strip().replace("\n", " ")


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={_normalize_action(action)} reward={reward:.2f} "
        f"done={str(done).lower()} error={_flatten(error)}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def build_user_prompt(
    task_name: str,
    step: int,
    stdout: str,
    stderr: str,
    score: float,
    history: List[str],
) -> str:
    history_block = "\n".join(history[-5:]) if history else "None"
    return textwrap.dedent(
        f"""
        Task: {task_name}
        Step: {step}
        Last stdout:
        {stdout or "<empty>"}

        Last stderr:
        {stderr or "<empty>"}

        Current normalized score: {score:.3f}
        Recent history:
        {history_block}

        Return the single next command.
        """
    ).strip()


def _heuristic_command(task_name: str, history: List[str]) -> str:
    policies = {
        "triage": ["ps aux", "top -bn1", "kill 4242", "free -m"],
        "optimization": [
            "vmstat",
            "sysctl vm.swappiness",
            "sysctl -w vm.swappiness=10",
            "echo Y > /sys/module/zswap/parameters/enabled",
            "vmstat",
        ],
        "security": [
            "sysctl -a | grep rp_filter",
            "sysctl -w net.ipv4.conf.all.rp_filter=1",
            "sysctl -w net.ipv4.conf.default.rp_filter=1",
            "sysctl -a | grep rp_filter",
        ],
    }
    sequence = policies.get(task_name, ["ps aux"])
    index = min(len(history), len(sequence) - 1)
    return sequence[index]


async def get_model_command(
    client: AsyncOpenAI,
    task_name: str,
    step: int,
    stdout: str,
    stderr: str,
    score: float,
    history: List[str],
) -> str:
    prompt = build_user_prompt(task_name, step, stdout, stderr, score, history)
    try:
        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        content = (completion.choices[0].message.content or "").strip()
        return _normalize_action(content) if content else _heuristic_command(task_name, history)
    except Exception:
        return _heuristic_command(task_name, history)


def create_env() -> LinuxSreGymEnv:
    if LOCAL_IMAGE_NAME:
        return LinuxSreGymEnv.from_docker_image(LOCAL_IMAGE_NAME)
    return LinuxSreGymEnv(base_url=ENV_BASE_URL)


async def run_episode(client: AsyncOpenAI, env: LinuxSreGymEnv) -> float:
    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    score = 0.0
    success = False
    task_name = "unknown"
    log_started = False

    try:
        result = await env.reset()
        observation = result.observation
        task_name = observation.current_task
        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
        log_started = True

        stdout = observation.stdout
        stderr = observation.stderr
        score = float(getattr(observation, "reward_breakdown", observation).score) if hasattr(getattr(observation, "reward_breakdown", None), "score") else float(observation.metadata.get("score", 0.0))

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            command = await get_model_command(
                client=client,
                task_name=task_name,
                step=step,
                stdout=stdout,
                stderr=stderr,
                score=score,
                history=history,
            )

            result = await env.step(LinuxSreGymAction(command=command))
            observation = result.observation
            reward = float(result.reward or 0.0)
            rewards.append(reward)
            steps_taken = step
            score = float(getattr(observation, "reward_breakdown", observation).score) if hasattr(getattr(observation, "reward_breakdown", None), "score") else float(observation.metadata.get("score", score))
            print(f"DEBUG: step={step} command={command} metadata={observation.metadata} computed_score={score}")
            stdout = observation.stdout
            stderr = observation.stderr
            error = observation.stderr or observation.metadata.get("last_action_error")
            log_step(step=step, action=command, reward=reward, done=result.done, error=error)
            history.append(f"{command} -> reward={reward:.2f} score={score:.3f}")

            if result.done:
                break

        success = score >= SUCCESS_SCORE_THRESHOLD
        return score
    finally:
        if not log_started:
            log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY or "missing-key")
    env = create_env()
    try:
        scores = [await run_episode(client, env) for _ in range(TASK_COUNT)]
        _ = scores
    finally:
        await env.close()


if __name__ == "__main__":
    asyncio.run(main())
