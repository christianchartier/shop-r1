# Commands Ledger — GRPO Dual‑Server (Paper‑Faithful)

Use these exact, line‑by‑line commands to bring up the two vLLM servers and run GRPO. Keep tmux commands on one line (no embedded newlines).

## 0) Preflight (any terminal)
```
cd /workspace/shop-r1
source .venv/bin/activate
python -m pip install --no-cache-dir "vllm==0.10.1.1" wandb
```

## 1) Create a small RL dataset
```
python environments/shop_r1/synthesize.py -o data/rl.jsonl -n 200 --seed 7
```

## 2) Terminal A (tmux) — Start TRL vLLM communicator (GPU 0, port 8000)
```
tmux kill-session -t vllm_trl || true
tmux new -d -s vllm_trl "cd /workspace/shop-r1 && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=0 python -m trl.scripts.vllm_serve --model Qwen/Qwen2.5-0.5B-Instruct --host 0.0.0.0 --port 8000 --max-model-len 1024 --gpu-memory-utilization 0.60"
# Verify TRL endpoints (should be HTTP/1.1 200)
curl -s -i http://localhost:8000/get_world_size/
```

## 3) Terminal B (tmux) — Start OpenAI vLLM server (GPU 1, port 8001)
```
tmux kill-session -t vllm_oai || true
tmux new -d -s vllm_oai "cd /workspace/shop-r1 && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-0.5B-Instruct --host 0.0.0.0 --port 8001 --dtype auto --max-model-len 1024 --gpu-memory-utilization 0.50 --disable-log-requests --enforce-eager --max-num-batched-tokens 512"
# Verify OpenAI routes
curl -s http://localhost:8001/v1/models | jq .
```

## 4) Terminal C — Run GRPO trainer (GPU 1)
```
cd /workspace/shop-r1 && source .venv/bin/activate
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://localhost:8001/v1
CUDA_VISIBLE_DEVICES=1 python scripts/rl_train_grpo.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset data/rl.jsonl \
  --output_dir checkpoints/rl_shop_r1 \
  --strict --sim_threshold 0.75 \
  --alpha 0.005 --beta 0.001 --dars_factor 1000 \
  --temperature 0.6 \
  --per_device_batch_size 1 --num_generations 8 --grad_accum 8 \
  --max_steps 50 --save_steps 25 --eval_steps 25 \
  --max_seq_len 1024 --learning_rate 1e-7
```

## 5) Monitoring
```
tmux ls
tmux capture-pane -pt vllm_trl | tail -n 60
tmux capture-pane -pt vllm_oai | tail -n 60
watch -n 1 nvidia-smi
```

## 6) Cleanup
```
tmux kill-session -t vllm_trl || true
tmux kill-session -t vllm_oai || true
```

## GPU notes
- CUDA_VISIBLE_DEVICES=0 targets physical GPU 0; =1 targets GPU 1.
- Keep TRL communicator (port 8000) and OpenAI server (port 8001) on different GPUs to avoid memory contention.
- The trainer uses OpenAI routes on 8001; TRL endpoints on 8000 are used internally by the trainer.

