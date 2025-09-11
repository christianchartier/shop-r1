#!/usr/bin/env python3
"""
Multi‑turn evaluation: iterates episodes with steps, queries a model per step,
parses action JSON, and computes paper metrics across all steps.

Input dataset: JSONL with one episode per line:
  {"steps": [ {"prompt": [... or str], "answer": {type,name,text}}, ... ]}

If "steps" is absent, each line is treated as a 1‑step episode with fields
"prompt" and "answer".
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI


# Make sure we can import local modules
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from configs.endpoints import ENDPOINTS  # noqa: E402
from scripts.eval_paper_metrics import PaperMetricsEvaluator  # noqa: E402


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
        for i, line in enumerate(f):
            if max_episodes > 0 and len(episodes) >= max_episodes:
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


def extract_prompt_text(step: Dict[str, Any]) -> str:
    p = step.get("prompt")
    if isinstance(p, str):
        return p
    if isinstance(p, list) and p:
        # use first user message content
        for msg in p:
            if isinstance(msg, dict) and msg.get("role") == "user":
                c = msg.get("content")
                if isinstance(c, str):
                    return c
    return ""


def parse_action_from_output(text: str) -> Dict[str, Any] | None:
    # Find a JSON object in the text
    matches = re.findall(r"\{[\s\S]*?\}", text)
    for m in matches:
        try:
            obj = json.loads(m)
        except Exception:
            continue
        # Accept nested { rationale, action: {type,name,text} }
        if isinstance(obj, dict) and "action" in obj and isinstance(obj["action"], dict):
            act = obj["action"]
            # Fill missing fields
            act.setdefault("name", "")
            act.setdefault("text", "")
            return act
        # Or flattened { rationale, type, name, text }
        if isinstance(obj, dict) and "type" in obj:
            obj.setdefault("name", "")
            obj.setdefault("text", "")
            return {"type": obj.get("type"), "name": obj.get("name"), "text": obj.get("text")}
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi‑turn evaluation for Shop‑R1 episodes")
    ap.add_argument("--dataset", required=True, help="Path to episodes JSONL")
    ap.add_argument("--max_episodes", type=int, default=-1, help="Max episodes to evaluate")
    ap.add_argument("--model_alias", default="local-qwen", help="Endpoint alias (configs/endpoints.py)")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--sim_threshold", type=float, default=0.75)
    ap.add_argument("--output", default="results/evaluation/multiturn_eval.json")

    args = ap.parse_args()

    # Load episodes
    episodes = load_episodes(args.dataset, args.max_episodes)
    if not episodes:
        print("No episodes found.")
        return 1

    # Endpoint selection
    if args.model_alias not in ENDPOINTS:
        print(f"Unknown model alias '{args.model_alias}'. Available: {', '.join(ENDPOINTS.keys())}")
        return 2
    ep = ENDPOINTS[args.model_alias]
    client = OpenAI(api_key=os.getenv(ep["key"], "EMPTY"), base_url=ep["url"]) 
    model = ep["model"]

    evaluator = PaperMetricsEvaluator(sim_threshold=args.sim_threshold)

    truths: List[Dict[str, Any]] = []
    preds: List[Dict[str, Any]] = []

    total_steps = 0
    for epi_idx, epi in enumerate(episodes):
        steps = epi.get("steps", [])
        for t, step in enumerate(steps):
            total_steps += 1
            prompt_text = extract_prompt_text(step)
            # Build improved prompt
            user_prompt = f"{IMPROVED_INSTRUCTION}\n\n{prompt_text}\n\nJSON Action:"
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
                print(f"[warn] episode {epi_idx} step {t}: request failed: {e}")
                output = ""

            pred_action = parse_action_from_output(output) or {}
            # Ground truth action
            gt = step.get("answer")
            if isinstance(gt, str):
                try:
                    gt = json.loads(gt)
                except Exception:
                    gt = {}
            if not isinstance(gt, dict):
                gt = {}
            truths.append(gt)
            preds.append(pred_action)

    metrics = evaluator.compute_metrics(truths, preds)
    print("\n=== MULTI‑TURN EVALUATION (per‑step aggregate) ===")
    print(f"Episodes: {len(episodes)}  Steps: {total_steps}")
    print(f"Exact Action Accuracy:  {metrics.exact_action_acc:.2%}")
    print(f"Action Type Accuracy:   {metrics.action_type_acc:.2%}")
    print(f"Action Type F1 (Macro): {metrics.action_type_f1:.2%}")

    # Save JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = metrics.to_dict()
    payload.update({"episodes": len(episodes), "total_steps": total_steps})
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved results to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

