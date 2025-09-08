# Shop‑R1 Environment

## Overview
- Environment ID: `shop_r1`
- Type: single-turn and multi-turn (paper-aligned)
- Output schema: a single JSON object with keys `rationale` (string) and `action` (object with `type`, `name`, `text`). Allowed `type`: `click`, `type_and_submit`, `terminate`.
- Rewards: hierarchical with Difficulty‑Aware Reward Scaling (DARS) and optional self‑certainty on the rationale.

## Install
- UV: `uv pip install -e environments/shop_r1`
- Pip: `pip install -e environments/shop_r1`

Optional training extras: `pip install -e environments/shop_r1[train]`

## Quickstart (Local, Offline)
1) Synthesize a tiny dataset
```
python environments/shop_r1/synthesize.py -o data/shop_r1_tiny.jsonl -n 20 --seed 42
```
2) Run an evaluation
- Via vf-eval (if auto-discovery is enabled in this repo):
```
uv run vf-eval shop_r1 -n 20 -a '{"dataset_path":"data/shop_r1_tiny.jsonl"}'
```
- Or via a Python script in this repo:
```
python scripts/eval_paper_metrics.py \
  --dataset data/shop_r1_tiny.jsonl \
  --max_examples 20 \
  --output results/shop_r1_results.json
```

## Action JSON Schema
Example outputs:
```
{"rationale": "Search for laptops.",
 "action": {"type": "type_and_submit", "name": "search_input", "text": "laptop"}}

{"rationale": "Open details.",
 "action": {"type": "click", "name": "view_details", "text": ""}}

{"rationale": "Done.",
 "action": {"type": "terminate", "name": "", "text": ""}}
```

## Rewards (Paper Mapping)
- Format: +0.5 if strict single‑JSON (no prose, fences, etc.).
- Action type: +0.3 exact‑match on `type`.
- Attribute presence:
  - click: +0.2 if `name` present
  - type_and_submit: +0.1 if `name` present; +0.1 if `text` present
- Value similarity:
  - ROUGE‑L(name/text) with threshold 0.75
  - DARS factor scales long‑text similarity (configurable)
- Optional self‑certainty (rationale): ~0.13 weight; derived from token‑level logprobs when available.

Key args to `load_environment(...)` / `load_multiturn_environment(...)`:
- `dataset_path`, `max_examples` | multi‑turn accepts trajectory JSONL (see below)
- `strict`: enforce schema strictly (recommended for eval)
- `enable_dars`, `dars_*` weights; `sim_threshold`
- `enable_self_certainty`, `w_self_certainty`, `sc_*` calibration

## Dataset Formats
- Single‑turn JSONL (one example per line):
```
{"prompt": [{"role": "user", "content": "...HTML/context..."}],
 "answer": {"type": "click", "name": "add_to_cart"}}
```

- Multi‑turn JSONL (one episode per line):
```
{"steps": [
  {"prompt": [{"role": "user", "content": "context t1"}],
   "answer": {"type": "type_and_submit", "name": "search_input", "text": "laptop"}},
  {"prompt": [{"role": "user", "content": "context t2"}],
   "answer": {"type": "click", "name": "view_details"}}
]}
```
If `steps` is absent, the loader treats each line as a 1‑step episode.

## System Prompt (Paper)
Use this for evaluation to match the paper’s constraints:
```
<IMPORTANT>
Your task is to predict the next action and provide rationale for the action based
on the previous actions and context.
You need to pretend that you are a user, browsing amazon.com and searching for a
product to purchase.
The history action (with details described below) and context will be provided to
you.
You need to predict the next action and provide rationale for the action.
</IMPORTANT>
# Action Space
An action is represented in JSON format, and there are three primary types of actions:
1) type_and_submit {"type":"type_and_submit","name":"input_name","text":"search_text"}
2) click {"type":"click","name":"clickable_name"}
3) terminate {"type":"terminate"}
# Output Format
Output a single JSON object with keys rationale and action (no extra text).
<IMPORTANT> OUTPUT A SINGLE JSON OBJECT, NOTHING ELSE. </IMPORTANT>
```

## Optional: RunPod Setup
For full SFT/RL training and vLLM serving on a pod, see `docs/runpod.md`.

## Citation
Zhang et al., Shop‑R1 (arXiv:2507.17842)

