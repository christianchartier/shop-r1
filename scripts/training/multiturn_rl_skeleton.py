#!/usr/bin/env python3
"""
Minimal multi‑turn RL skeleton for Shop‑R1.

Iterates multi‑turn episodes (steps), queries a model per step with improved
prompting (+optional whole‑session context), computes rewards via the
Shop‑R1 rubric (strict JSON gating), and logs per‑episode returns.

This is a scaffolding script: it does not update model weights. Integrate with
your trainer of choice to apply policy updates using these per‑step rewards.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

from environments.shop_r1.shop_r1 import load_environment


IMPROVED_INSTRUCTION = (
    "You are a web navigation assistant. Respond with a JSON action.\n\n"
    "IMPORTANT: Use ONLY these action types:\n"
    "- click: Click an element (requires name field)\n"
    "- type_and_submit: Type text and submit (requires name and text)\n"
    "- terminate: End the task (no extra fields)\n\n"
    "Respond with EXACTLY this shape (no extra keys, no prose):\n"
    "{\n  \"rationale\": \"why this action\",\n  \"type\": \"click|type_and_submit|terminate\",\n  \"name\": \"element name (empty for terminate)\",\n  \"text\": \"text to type (empty except type_and_submit)\"\n}\n"
)


def load_episodes(path: str, max_episodes: int = -1) -> List[Dict[str, Any]]:
    episodes: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if 0 < max_episodes <= len(episodes):
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("steps"), list):
                episodes.append(obj)
            elif isinstance(obj, dict):
                step = {"prompt": obj.get("prompt"), "answer": obj.get("answer")}
                episodes.append({"steps": [step]})
    return episodes


def extract_prompt(step: Dict[str, Any]) -> str:
    p = step.get("prompt")
    if isinstance(p, str):
        return p
    if isinstance(p, list):
        for msg in p:
            if isinstance(msg, dict) and msg.get("role") == "user" and isinstance(msg.get("content"), str):
                return msg["content"]
    return ""


def action_summary(a: Dict[str, Any]) -> str:
    t = str(a.get("type", "")).lower()
    name = a.get("name") or ""
    text = a.get("text") or ""
    if t == "click":
        return f"click({name})"
    if t == "type_and_submit":
        return f"type_and_submit({name}='{text}')"
    if t == "terminate":
        return "terminate()"
    return t or "action()"


def main() -> int:
    ap = argparse.ArgumentParser(description="Minimal multi‑turn RL skeleton (no weight updates)")
    ap.add_argument("--dataset", required=True, help="Episodes JSONL")
    ap.add_argument("--max_episodes", type=int, default=10)
    ap.add_argument("--model_alias", default="local-qwen")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--sim_threshold", type=float, default=0.75)
    ap.add_argument("--whole_session", action="store_true")
    ap.add_argument("--output", default="results/evaluation/multiturn_rl_skeleton.json")
    args = ap.parse_args()

    # Endpoint
    from configs.endpoints import ENDPOINTS
    if args.model_alias not in ENDPOINTS:
        print(f"Unknown model alias '{args.model_alias}'")
        return 2
    ep = ENDPOINTS[args.model_alias]
    client = OpenAI(api_key=os.getenv(ep["key"], "EMPTY"), base_url=ep["url"]) 
    model = ep["model"]

    episodes = load_episodes(args.dataset, args.max_episodes)
    if not episodes:
        print("No episodes loaded")
        return 1

    # Build a tiny env only to reuse parser + rubric (rewards)
    env = load_environment(strict=args.strict, sim_threshold=args.sim_threshold)
    parser = env.parser
    rubric = env.rubric

    results: List[Dict[str, Any]] = []
    for ei, epi in enumerate(episodes):
        steps = epi.get("steps", [])
        ep_return = 0.0
        pred_actions: List[Dict[str, Any]] = []
        for ti, step in enumerate(steps):
            prompt = extract_prompt(step)
            if args.whole_session and pred_actions:
                hist = ", ".join(action_summary(a) for a in pred_actions)
                prompt = (prompt.rstrip() + f"\nPrevious actions: {hist}.") if prompt else f"Previous actions: {hist}."
            user_prompt = f"{IMPROVED_INSTRUCTION}\n\n{prompt}\n\nJSON Action:"
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=args.temperature,
                    max_tokens=160,
                    response_format={"type": "json_object"},
                )
                output = resp.choices[0].message.content or ""
            except Exception as e:
                print(f"[warn] episode {ei} step {ti} request failed: {e}")
                output = ""

            # Compute reward with rubric
            gt = step.get("answer") if isinstance(step.get("answer"), dict) else {}
            extras = {"info": gt, "prompt": step.get("prompt")}
            step_reward = 0.0
            for w, fn in zip(getattr(rubric, "weights", []), getattr(rubric, "funcs", [])):
                try:
                    step_reward += float(w) * float(fn(output, {}, **extras))
                except Exception:
                    continue
            # Accumulate
            ep_return += step_reward
            # Record predicted action JSON for next step history
            try:
                act = parser.parse_answer(output) or {}
            except Exception:
                act = {}
            pred_actions.append(act)
            # Terminate if model says so
            if act.get("type") == "terminate":
                break

        results.append({"episode": ei, "return": ep_return, "steps": len(pred_actions)})
        print(f"episode {ei}: return={ep_return:.4f}, steps={len(pred_actions)}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)
    print(f"Saved skeleton results to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

