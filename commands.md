# Step‑By‑Step Commands (with Terminal labels)

Use these one‑liners in the specified terminal. Replace the IP/PORT with your instance values.

## Terminal C (Remote A100) — vLLM server and training

0) Authenticate with GitHub and clone into /ephemeral (run once).

apt-get update && apt-get install -y gh

# Device login; if browser can't open, paste the URL manually
gh auth login -s repo -w

# Configure git to use gh for credentials (prevents username/password prompts)
gh auth setup-git || git config --global credential.helper '!gh auth git-credential'

# Clone the repo into /ephemeral and enter it
gh repo clone christianchartier/shop-r1 /ephemeral/shop-r1 && cd /ephemeral/shop-r1

1) Start vLLM server (dedicate this terminal). If it’s already running, skip.

# Option A (recommended): run inside tmux so this terminal can be reused
apt-get install -y tmux && tmux new -s vllm
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-3B-Instruct --host 0.0.0.0 --port 8000 --dtype auto --max-model-len 32768 --gpu-memory-utilization 0.90
# Detach with: Ctrl-b then d. Reattach later with: tmux attach -t vllm. Stop with Ctrl-C inside the tmux session.

# Option B: run in background and tail the log
nohup python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-3B-Instruct --host 0.0.0.0 --port 8000 --dtype auto --max-model-len 32768 --gpu-memory-utilization 0.90 > vllm.log 2>&1 &
tail -f vllm.log
# Stop background server: pkill -f "vllm.entrypoints.openai.api_server.*Qwen/Qwen2.5-3B-Instruct"

2) Open a second SSH session for interactive work (Terminal C2) and initialize the project (run once). Note: STOP vLLM before SFT so GPU is free.

# On your Mac, open a new terminal/tab and connect:
ssh -p 1234 root@69.19.136.229

# Sync latest code (pull updated training scripts) in Terminal C2.
git -C /ephemeral/shop-r1 pull --rebase origin main && cd /ephemeral/shop-r1

python3 -m venv .venv && source .venv/bin/activate && python -m pip install -U pip && python -m pip install verifiers datasets requests transformers accelerate peft && python -m pip install -e . && vf-install shop-r1

3) Synthesize a small dataset (run on remote; creates data/sft.jsonl).

python environments/shop_r1/synthesize.py -o data/sft.jsonl -n 1000 --seed 7

3a) Verify dataset file exists and looks correct (line count + first row).

echo -n "rows: " && wc -l < data/sft.jsonl && echo "first row:" && head -n 1 data/sft.jsonl

4) SFT training (run in Terminal C2; ensure vLLM is STOPPED).

python scripts/sft_train.py --dataset data/sft.jsonl --model Qwen/Qwen2.5-3B-Instruct --output_dir checkpoints/sft --epochs 4 --lr 2e-5 --per_device_batch_size 1 --grad_accum 64 --max_seq_len 32768

4a) Verify SFT checkpoint artifacts exist (quick listing).

echo "SFT checkpoint files:" && ls -lh checkpoints/sft | head -n 20

5) Restart vLLM server after SFT (Terminal C tmux/background; start it again as in step 1).

python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-3B-Instruct --host 0.0.0.0 --port 8000 --dtype auto --max-model-len 32768 --gpu-memory-utilization 0.90

5a) Check vLLM health (Terminal C2 — any remote shell). Note: some vLLM versions return an empty body; check HTTP 200.

curl -s -o /dev/null -w "HTTP %{http_code}\n" http://localhost:8000/health

5b) List served models (Terminal C2) to confirm API JSON is reachable.

# If jq is available:
curl -s http://localhost:8000/v1/models | jq .

# If jq is not installed, install it:
apt-get update && apt-get install -y jq

# Or, without installing jq, pretty-print via Python:
curl -s http://localhost:8000/v1/models | python -m json.tool | head -n 40

6) GRPO training (RL) — Terminal C2 (Remote A100). Keep vLLM running in Terminal C tmux.

python scripts/rl_train_grpo.py --model checkpoints/sft --dataset data/sft.jsonl --output_dir checkpoints/rl_shop_r1 --alpha 0.13 --beta 0.001 --dars_factor 1000 --max_steps 500 --learning_rate 1e-7 --temperature 0.6 --per_device_batch_size 1 --num_generations 8 --grad_accum 8

6a) If RL errors about missing verifiers APIs, upgrade verifiers to the latest from GitHub (Terminal C2), then retry step 6.

python -c "import importlib,sys;vf=importlib.import_module('verifiers');ok=all(hasattr(vf,a) for a in ('GRPOTrainer','load_environment'));print('verifiers_ok',ok);sys.exit(0 if ok else 1)" || python -m pip install -U 'verifiers @ git+https://github.com/willccbb/verifiers@main'

6b) If the verifiers upgrade fails due to Python 3.10, install Python 3.11 and use a new venv (Terminal C2), then retry step 6.

# Install Python 3.11 (try default repo first; if it fails, add deadsnakes PPA)
apt-get update && apt-get install -y python3.11 python3.11-venv || (apt-get update && apt-get install -y software-properties-common && add-apt-repository -y ppa:deadsnakes/ppa && apt-get update && apt-get install -y python3.11 python3.11-venv)

# Create and activate a 3.11 venv, reinstall deps and the project
python3.11 -m venv .venv311 && source .venv311/bin/activate && python -V && python -m pip install -U pip && python -m pip install 'verifiers @ git+https://github.com/willccbb/verifiers@main' transformers accelerate peft datasets requests && python -m pip install -e .

# Run RL under Python 3.11 venv
python scripts/rl_train_grpo.py --model checkpoints/sft --dataset data/sft.jsonl --output_dir checkpoints/rl_shop_r1 --alpha 0.13 --beta 0.001 --dars_factor 1000 --max_steps 500 --learning_rate 1e-7 --temperature 0.6 --per_device_batch_size 1 --num_generations 8 --grad_accum 8

7) Paper metrics (exact‑match, type accuracy, macro‑F1) — Terminal C2 (Remote A100). Targets the remote vLLM at localhost:8000.

python scripts/eval_actions.py --dataset data/sft.jsonl --model_alias local-qwen --sim_threshold 0.75 --out eval_results.json

8) Component ablations (short GRPO runs) — run on remote.

python scripts/ablate_components.py --model Qwen/Qwen2.5-3B-Instruct --dataset data/sft.jsonl --max_steps 100 --num_generations 4 --grad_accum 4 --out ablations.json

## Terminal B (Local Mac) — SSH port forward (keep open)

1) Open the port forward to the A100 (run and keep this window open during eval/RL).

ssh -p 1234 -N -L 8000:localhost:8000 root@69.19.136.229 -v

If this disconnects, re‑run it. Terminal A evals will fail with “Connection error” until this is up.

## Terminal A (Local Mac) — Evaluation against remote vLLM

1) Point to the forwarded vLLM endpoint (run once per shell).

export OPENAI_API_KEY=EMPTY && export OPENAI_BASE_URL=http://localhost:8000/v1

1a) Check vLLM health endpoint over the port forward.

curl -s "$OPENAI_BASE_URL/health" && echo

2) Non‑strict evaluation (normalization ON; good for sanity checks).

vf-eval shop-r1 -m local-qwen -a '{"strict":false,"normalize_variants":true,"sim_threshold":0.75,"debug_rewards":true,"debug_logprobs":true,"w_self_certainty":0.13}' -S '{"logprobs":true,"top_logprobs":5,"temperature":0.2,"response_format":{"type":"json_object"}}' -n 2 -r 1 -v

3) Strict evaluation (paper‑faithful format; JSON must be exact).

vf-eval shop-r1 -m local-qwen -a '{"strict":true,"normalize_variants":false,"sim_threshold":0.75,"debug_rewards":true,"debug_logprobs":true,"w_self_certainty":0.13,"system_prompt":"Output only a single JSON object with exactly two top-level keys: rationale (string) and action (object with keys type, name, text). Allowed types: click, type_and_submit, terminate. Type rules: terminate -> name=\\\"\\\" and text=\\\"\\\"; click -> name!=\\\"\\\" and text=\\\"\\\"; type_and_submit -> name!=\\\"\\\" and text!=\\\"\\\". No markdown, no extra keys, no commentary."}' -S '{"logprobs":true,"top_logprobs":5,"temperature":0,"response_format":{"type":"json_object"}}' -n 2 -r 1 -v

Notes
- For SFT: STOP vLLM in Terminal C (Ctrl‑C) before starting SFT, then restart vLLM after SFT.
- For “Connection error” in Terminal A: restart the SSH port forward in Terminal B and retry.
- Keep JSON in -a compact (no stray spaces/line breaks in keys) to avoid parse errors.
