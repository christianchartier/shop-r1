# PR: Add Shop‑R1 Environment (single‑turn + multi‑turn)

## Summary
Adds a paper‑aligned Shop‑R1 environment for simulating human behavior in online shopping. The environment outputs a single JSON object with a rationale and an action (`type`, `name`, `text`). Rewards follow the hierarchical scheme with Difficulty‑Aware Reward Scaling (DARS) and an optional self‑certainty signal for the rationale.

- ID: `shop_r1`
- Location: `environments/shop_r1/`
- Interfaces:
  - `load_environment(**kwargs)` (single‑turn)
  - `load_multiturn_environment(**kwargs)` (multi‑turn; requires `verifiers.MultiTurnEnv`)

## Install
- `uv pip install -e environments/shop_r1`
- Optional training extras: `pip install -e environments/shop_r1[train]`

## Quickstart
```
python environments/shop_r1/synthesize.py -o data/shop_r1_tiny.jsonl -n 20 --seed 42
uv run vf-eval shop_r1 -n 20 -a '{"dataset_path":"data/shop_r1_tiny.jsonl"}'
```
Fallback:
```
python scripts/eval_paper_metrics.py --dataset data/shop_r1_tiny.jsonl --max_examples 20 --output results/shop_r1_results.json
```

## Dataset
- Single‑turn JSONL with `prompt` and `answer`.
- Multi‑turn JSONL with `steps: [{prompt, answer}, ...]`. If `steps` is absent, each line is treated as a 1‑step episode.

## Rewards (Paper mapping)
- Format +0.5 (strict JSON only)
- Action type +0.3 exact‑match
- Attribute presence: click +0.2 name; type_and_submit +0.1 name +0.1 text
- Value similarity via ROUGE‑L with threshold 0.75; long‑text scaled by DARS
- Optional self‑certainty on rationale (~0.13 weight)

## System Prompt
Appendix‑style strict prompt included in `environments/shop_r1/README.md`.

## Tests
- `tests/test_shop_r1_smoke.py` (offline):
  - Verifies format gating, type reward behavior, and MultiTurn constructor if available.

## Optional: RunPod
- One‑shot pod setup and GRPO details in `docs/runpod.md`.

## Checklist
- [x] Env folder under `environments/shop_r1/` with README and pyproject
- [x] Single‑turn and multi‑turn constructors
- [x] Paper‑aligned rewards with DARS + self‑certainty
- [x] Tiny dataset synth and schema docs
- [x] Offline smoke test added
- [x] Optional RunPod doc, scripts referenced by in‑repo paths

## Known gaps / future work
- Self‑certainty exact KL(p||Uniform) estimator uses top‑k fallback when full logits unavailable; extend providers for full logprobs.
- Additional ablations & temperature sweep scripts to mirror paper figures.
- Multi‑turn evaluation metrics (session summaries) can be expanded per hub conventions.

