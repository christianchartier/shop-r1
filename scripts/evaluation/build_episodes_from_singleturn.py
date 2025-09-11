#!/usr/bin/env python3
"""
Build multi‑turn episodes from a single‑turn JSONL dataset.

Input format (single‑turn JSONL):
  {"prompt": [... or str], "answer": {"type":..., "name":..., "text":...}}

Output format (episodes JSONL):
  {"steps": [ {"prompt": ..., "answer": {...}}, ... ]}

Options allow fixed window sizes (chunk/slide) or random grouping. You can also
prepend a compact history summary of previous actions into each step's prompt.
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Any, Dict, List


def load_singleturn(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict) and ("prompt" in obj or "answer" in obj):
                rows.append(obj)
    return rows


def action_to_str(a: Dict[str, Any]) -> str:
    t = str(a.get("type", "")).lower()
    name = a.get("name") or a.get("target") or ""
    text = a.get("text") or a.get("value") or ""
    if t == "click":
        return f"click({name})" if name else "click(…)"
    if t == "type_and_submit":
        n = f"{name}" if name else "…"
        x = f"{text}" if text else "…"
        return f"type_and_submit({n}='{x}')"
    if t == "terminate":
        return "terminate()"
    return t or "action(…)"


def add_history_to_step(step: Dict[str, Any], prev_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not prev_actions:
        return step
    summary = ", ".join(action_to_str(a) for a in prev_actions)
    history_line = f"Previous actions: {summary}."
    p = step.get("prompt")
    if isinstance(p, str):
        step["prompt"] = p.rstrip() + "\n" + history_line
        return step
    if isinstance(p, list) and p:
        # append to first user message or create one
        # find first user message
        for msg in p:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content") or ""
                if isinstance(content, str):
                    msg["content"] = content.rstrip() + "\n" + history_line
                    return step
        # otherwise, append a user message
        p.append({"role": "user", "content": history_line})
        return step
    # fallback: just set a string prompt
    step["prompt"] = history_line
    return step


def build_episodes(
    rows: List[Dict[str, Any]],
    steps: int,
    mode: str,
    max_episodes: int,
    include_history: bool,
    seed: int | None,
) -> List[Dict[str, Any]]:
    n = len(rows)
    idxs: List[int] = list(range(n))
    if seed is not None:
        random.seed(seed)
    episodes: List[Dict[str, Any]] = []

    def mk_episode(indices: List[int]) -> Dict[str, Any] | None:
        steps_list: List[Dict[str, Any]] = []
        prev_actions: List[Dict[str, Any]] = []
        for k in indices:
            r = rows[k]
            step = {"prompt": r.get("prompt"), "answer": r.get("answer")}
            if include_history:
                step = add_history_to_step(step, prev_actions)
            # update history
            a = r.get("answer")
            if isinstance(a, dict):
                prev_actions.append(a)
            steps_list.append(step)
        return {"steps": steps_list}

    if mode == "chunk":
        for i in range(0, n, steps):
            block = idxs[i : i + steps]
            if len(block) == steps:
                episodes.append(mk_episode(block))
                if 0 < max_episodes <= len(episodes):
                    break
    elif mode == "slide":
        for i in range(0, n - steps + 1):
            block = idxs[i : i + steps]
            episodes.append(mk_episode(block))
            if 0 < max_episodes <= len(episodes):
                break
    elif mode == "random":
        starts = list(range(0, n - steps + 1))
        random.shuffle(starts)
        for s in starts:
            block = idxs[s : s + steps]
            episodes.append(mk_episode(block))
            if 0 < max_episodes <= len(episodes):
                break
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return episodes


def main() -> int:
    ap = argparse.ArgumentParser(description="Build episodes JSONL from single‑turn JSONL")
    ap.add_argument("--input", required=True, help="Path to single‑turn JSONL")
    ap.add_argument("--output", required=True, help="Path to write episodes JSONL")
    ap.add_argument("--steps", type=int, default=3, help="Steps per episode")
    ap.add_argument("--mode", choices=["chunk", "slide", "random"], default="chunk")
    ap.add_argument("--max_episodes", type=int, default=-1)
    ap.add_argument("--include-history", action="store_true", help="Prepend history summary to each step prompt")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    rows = load_singleturn(args.input)
    if not rows:
        print("No rows found in input.")
        return 1
    episodes = build_episodes(
        rows=rows,
        steps=args.steps,
        mode=args.mode,
        max_episodes=args.max_episodes,
        include_history=args.include_history,
        seed=args.seed,
    )
    with open(args.output, "w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")
    print(f"Wrote {len(episodes)} episodes to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

