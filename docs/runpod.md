# RunPod One‑Shot Setup (Shop‑R1)

This document mirrors the working instructions from FRESH_POD_SETUP.md, adapted to repo‑local scripts and paths. It enables a one‑shot setup for SFT, GRPO and evaluation with vLLM servers.

## 1) SSH into pod
```
ssh -p 1234 root@[YOUR_POD_IP]
```

## 2) Quick setup script
From the repository root on the pod:
```
cd /workspace
chmod +x deployment/RUNPOD_QUICK_SETUP.sh
./deployment/RUNPOD_QUICK_SETUP.sh
```
Notes:
- Pins Transformers/TRL to compatible versions and registers the environment.

## 3) Test SFT (tiny)
```
python scripts/sft_train.py \
  --dataset data/test.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output_dir checkpoints/test_sft \
  --epochs 1 \
  --save_steps 2 \
  --logging_steps 1 \
  --per_device_batch_size 1 \
  --grad_accum 1 \
  --max_seq_len 1024 \
  --lr 2e-5

ls -la checkpoints/test_sft/
```

## 4) GRPO Training (2 GPUs)
GRPO uses two vLLM servers:
1) TRL Communicator (port 8000, GPU 0)
2) OpenAI API Server (port 8001, GPU 1)

```
chmod +x scripts/training/run_grpo_complete.sh
./scripts/training/run_grpo_complete.sh --quick    # 1‑step quick test
./scripts/training/run_grpo_complete.sh            # ~50 steps
```
The script handles:
- vLLM + deps install
- RL dataset creation
- Server orchestration and cleanup
- Verified GRPO params

## 5) Evaluation
```
cd /workspace/shop-r1
source .venv/bin/activate

python environments/shop_r1/synthesize.py -o data/test.jsonl -n 100 --seed 42

./scripts/evaluation/run_evaluation_on_pod.sh

chmod +x scripts/evaluation/run_zero_shot_improved.sh
./scripts/evaluation/run_zero_shot_improved.sh data/test.jsonl 50
```
Key finding: with explicit action‑format instructions, the 0.5B model’s action type accuracy rises dramatically compared to naive zero‑shot prompting.

### Manual vLLM server (optional)
Terminal 1:
```
tmux new -s eval
cd /workspace/shop-r1 && source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 --port 8001 \
  --dtype auto \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.70 \
  --disable-log-requests \
  --enforce-eager
```
Terminal 2:
```
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://localhost:8001/v1
curl -s http://localhost:8001/v1/models | jq .

python scripts/eval_paper_metrics.py \
  --dataset data/test.jsonl \
  --model_alias local-qwen \
  --max_examples 50 \
  --output results/evaluation/zero_shot.json

python scripts/evaluation/fix_zero_shot_prompting.py \
  --dataset data/test.jsonl \
  --max_examples 50
```

## Troubleshooting
- Python 3.11 apt hang:
```
pkill -f apt
DEBIAN_FRONTEND=noninteractive apt-get install -y python3.11 python3.11-venv python3.11-dev
```

- TRL/Transformers conflicts:
```
pip uninstall -y transformers trl || true
pip install --no-cache-dir "transformers==4.56.1" "trl==0.21.0"
vf-install shop-r1
```

- vLLM OOM or startup issues: reduce memory and context length; e.g., `--max-model-len 512`, `--gpu-memory-utilization 0.50`, `--max-num-batched-tokens 512`.

## Notes
- The scripts here reference in‑repo paths; avoid external `wget`.
- For full details and context, see the paper and the environment README.

