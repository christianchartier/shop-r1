#!/usr/bin/env python3
"""
Multi‑turn evaluation using Transformers (HF) with full per‑token probability
vectors (exact softmax). Computes the same paper metrics (per‑step aggregate)
as the other evaluator, but generates locally via HF instead of an OpenAI API.

Usage example:
  uv run python scripts/evaluation/multiturn_eval_exact_kl.py \
    --dataset data/episodes.jsonl \
    --max_episodes 10 \
    --model_id Qwen/Qwen2.5-0.5B-Instruct \
    --device cuda \
    --whole_session \
    --output results/evaluation/multiturn_eval_exact.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from scripts.eval_paper_metrics import PaperMetricsEvaluator


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


def extract_prompt_text(step: Dict[str, Any]) -> str:
    p = step.get("prompt")
    if isinstance(p, str):
        return p
    if isinstance(p, list):
        for msg in p:
            if isinstance(msg, dict) and msg.get("role") == "user" and isinstance(msg.get("content"), str):
                return msg["content"]
    return ""


def build_history_summary(steps: List[Dict[str, Any]], upto_idx: int) -> str:
    if upto_idx <= 0:
        return ""
    parts: List[str] = []
    for s in steps[:upto_idx]:
        a = s.get("answer") or {}
        if not isinstance(a, dict):
            continue
        t = str(a.get("type", "")).lower()
        name = a.get("name") or ""
        text = a.get("text") or a.get("value") or ""
        if t == "click":
            parts.append(f"click({name})" if name else "click(…)")
        elif t == "type_and_submit":
            n, x = name or "…", text or "…"
            parts.append(f"type_and_submit({n}='{x}')")
        elif t == "terminate":
            parts.append("terminate()")
        elif t:
            parts.append(t)
    return ("Previous actions: " + ", ".join(parts) + ".") if parts else ""


def parse_action_from_output(text: str) -> Dict[str, Any] | None:
    matches = re.findall(r"\{[\s\S]*?\}", text)
    for m in matches:
        try:
            obj = json.loads(m)
        except Exception:
            continue
        if isinstance(obj, dict) and "action" in obj and isinstance(obj["action"], dict):
            act = obj["action"]
            act.setdefault("name", "")
            act.setdefault("text", "")
            return act
        if isinstance(obj, dict) and "type" in obj:
            obj.setdefault("name", "")
            obj.setdefault("text", "")
            return {"type": obj.get("type"), "name": obj.get("name"), "text": obj.get("text")}
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi‑turn eval with HF exact per‑token probabilities")
    ap.add_argument("--dataset", required=True, help="Episodes JSONL (or single‑turn JSONL)")
    ap.add_argument("--max_episodes", type=int, default=10)
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--whole_session", action="store_true")
    ap.add_argument("--include_prev_html", action="store_true")
    ap.add_argument("--html_max_chars", type=int, default=2000)
    ap.add_argument("--sim_threshold", type=float, default=0.75)
    ap.add_argument("--output", default="results/evaluation/multiturn_eval_exact.json")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype="auto", trust_remote_code=True).to(device).eval()
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    episodes = load_episodes(args.dataset, args.max_episodes)
    if not episodes:
        print("No episodes found.")
        return 1

    evaluator = PaperMetricsEvaluator(sim_threshold=args.sim_threshold)
    truths: List[Dict[str, Any]] = []
    preds: List[Dict[str, Any]] = []

    total_steps = 0
    for epi in episodes:
        steps = epi.get("steps", [])
        for t, step in enumerate(steps):
            total_steps += 1
            prompt_text = extract_prompt_text(step)
            if args.whole_session:
                hist = build_history_summary(steps, t)
                if hist:
                    prompt_text = (prompt_text.rstrip() + "\n" + hist) if prompt_text else hist
            if args.include_prev_html and t > 0:
                prev_htmls: List[str] = []
                for s in steps[:t]:
                    p = s.get("prompt")
                    if isinstance(p, list):
                        for msg in p:
                            if isinstance(msg, dict) and msg.get("role") == "user" and isinstance(msg.get("content"), str):
                                prev_htmls.append(msg["content"]) 
                    elif isinstance(p, str):
                        prev_htmls.append(p)
                if prev_htmls:
                    blob = "\n\n".join(prev_htmls)
                    if len(blob) > args.html_max_chars:
                        blob = blob[-args.html_max_chars:]
                    prompt_text = (prompt_text.rstrip() + "\n\nPrevious contexts (truncated):\n" + blob) if prompt_text else blob

            user_prompt = f"{IMPROVED_INSTRUCTION}\n\n{prompt_text}\n\nJSON Action:"
            enc = tok(user_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = mdl.generate(
                    **enc,
                    do_sample=True,
                    temperature=args.temperature,
                    max_new_tokens=160,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            # Compute per‑token full probabilities (softmax)
            probs_per_token: List[List[float]] = []
            for logits in out.scores:  # each tensor shape [1, vocab]
                p = torch.softmax(logits[0], dim=-1)
                # To avoid enormous JSON, we do not store these; but they are computed here.
                # If needed, one could compute exact‑KL now.
                # probs_per_token.append(p.tolist())
                pass
            # Decode completion and parse action
            gen_ids = out.sequences[0][enc.input_ids.size(1):]
            text = tok.decode(gen_ids, skip_special_tokens=True)
            pred = parse_action_from_output(text) or {}
            preds.append(pred)

            gt = step.get("answer")
            if isinstance(gt, str):
                try:
                    gt = json.loads(gt)
                except Exception:
                    gt = {}
            if not isinstance(gt, dict):
                gt = {}
            truths.append(gt)

    metrics = evaluator.compute_metrics(truths, preds)
    print("\n=== MULTI‑TURN EVALUATION (HF exact) ===")
    print(f"Episodes: {len(episodes)}  Steps: {total_steps}")
    print(f"Exact Action Accuracy:  {metrics.exact_action_acc:.2%}")
    print(f"Action Type Accuracy:   {metrics.action_type_acc:.2%}")
    print(f"Action Type F1 (Macro): {metrics.action_type_f1:.2%}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics.to_dict() | {"episodes": len(episodes), "total_steps": total_steps}, f, ensure_ascii=False, indent=2)
    print(f"Saved results to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

