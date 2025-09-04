# 🚀 FRESH POD SETUP - Complete Instructions

## Step 1: SSH into Fresh Pod
```bash
ssh -p 1234 root@62.169.159.61
```

## Step 2: Run Complete Setup (Copy & Paste This Entire Block)

```bash
# ============================================
# COMPLETE SETUP FOR FRESH POD - RUN ALL AT ONCE
# ============================================

# 1. Check environment and go to workspace
echo "=== Setting up Shop-R1 on Fresh Pod ==="
cd /workspace || cd /ephemeral || cd ~
pwd

# 2. Clone the repository
echo "=== Cloning repository ==="
git clone https://github.com/christianchartier/shop-r1.git
cd shop-r1

# 3. Create Python environment (try different versions)
echo "=== Creating Python environment ==="
if command -v python3.11 &> /dev/null; then
    python3.11 -m venv .venv
elif command -v python3.10 &> /dev/null; then
    python3.10 -m venv .venv
elif command -v python3.9 &> /dev/null; then
    python3.9 -m venv .venv
else
    python3 -m venv .venv
fi

# 4. Activate environment
source .venv/bin/activate
python --version

# 5. Upgrade pip and install core dependencies
echo "=== Installing dependencies (this takes 3-5 minutes) ==="
python -m pip install -U pip setuptools wheel

# Core ML packages
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip uninstall -y torchvision  # Remove if not needed

# Training packages
python -m pip install "transformers>=4.55,<5" "trl==0.21.0" "vllm==0.10.1.1"
python -m pip install accelerate>=0.30 peft>=0.11 datasets>=2.19 
python -m pip install requests openai

# Install verifiers
python -m pip install 'verifiers @ git+https://github.com/willccbb/verifiers@main'

# 6. Install shop-r1 environment
echo "=== Installing Shop-R1 environment ==="
python -m pip install -e .
vf-install shop-r1

# 7. Quick validation test
echo "=== Running quick test ==="
python tests/quick_test.py

# 8. Show GPU info
echo "=== GPU Information ==="
nvidia-smi -L || echo "No GPUs found"

echo "=== Setup Complete! ==="
echo "Next: Run testing commands below"
```

## Step 3: Create Test Data

```bash
# Create small test dataset
mkdir -p data
cat > data/test.jsonl << 'EOF'
{"prompt": [{"role": "user", "content": "Search for laptop"}], "answer": {"type": "type_and_submit", "name": "search", "text": "laptop"}, "rationale": "Looking for a laptop"}
{"prompt": [{"role": "user", "content": "Click add to cart"}], "answer": {"type": "click", "name": "add_to_cart"}, "rationale": "Adding to cart"}
{"prompt": [{"role": "user", "content": "Done shopping"}], "answer": {"type": "terminate"}, "rationale": "Finished"}
EOF

echo "Created test dataset with $(wc -l < data/test.jsonl) examples"
```

## Step 4: Test SFT Training (Quick Test)

```bash
# Run minimal SFT test
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

# Check results
ls -la checkpoints/test_sft/
```

## Step 5: Test GRPO (If You Have 2+ GPUs)

### Terminal 1: Start vLLM Server (in tmux)
```bash
# Create tmux session
tmux new -s vllm

# Inside tmux - run these commands:
cd /workspace/shop-r1 || cd /ephemeral/shop-r1
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python -m trl.scripts.vllm_serve \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.20

# Press Ctrl+B, then D to detach from tmux
```

### Terminal 2: Run GRPO Test (main terminal)
```bash
# Check server is running
sleep 15
curl -L -s -o /dev/null -w "Server status: %{http_code}\n" http://localhost:8000/get_world_size/

# Run GRPO
CUDA_VISIBLE_DEVICES=1 python scripts/rl_train_grpo.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset data/test.jsonl \
  --output_dir checkpoints/test_rl \
  --max_steps 2 \
  --save_steps 10 \
  --eval_steps 10 \
  --per_device_batch_size 1 \
  --num_generations 1 \
  --grad_accum 1 \
  --max_seq_len 1024 \
  --learning_rate 1e-7

# Kill vLLM server
tmux kill-session -t vllm
```

## Step 6: Run Evaluation

```bash
# Start evaluation server
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 --port 8001 \
  --dtype auto \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.90 &

SERVER_PID=$!
sleep 15

# Set environment and run eval
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://localhost:8001/v1

# Run evaluation
vf-eval shop-r1 -m Qwen/Qwen2.5-0.5B-Instruct -s -n 5

# Kill server
kill $SERVER_PID
```

## 📊 Expected Results

### ✅ Successful Setup
```
✓ Environment loaded
✓ Parser works
✓ Rewards computed
  Total reward: 0.8-1.0
```

### ✅ SFT Training
- Training loss decreases
- Saves checkpoint in `checkpoints/test_sft/`
- No error messages

### ✅ GRPO (if multi-GPU)
- Server responds with status 200
- Runs 2 training steps
- No UUID errors

### ✅ Evaluation
- Shows reward breakdown
- Action accuracy >0.3
- Format reward >0.8

## 🔧 Troubleshooting

### If Python 3.11 not found:
```bash
# Check available Python versions
ls /usr/bin/python3*

# Use the highest available (3.9+ required)
python3.10 -m venv .venv || python3.9 -m venv .venv
```

### If CUDA/GPU issues:
```bash
# Check CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check GPU memory
nvidia-smi
```

### If verifiers fails to install:
```bash
# Try installing from PyPI first
pip install verifiers

# Then install shop-r1
vf-install shop-r1
```

### Debug Reward Computation:
```bash
# Test rewards are being computed correctly
python -c "
import json
import verifiers as vf

env = vf.load_environment('shop-r1', debug_rewards=True)
completion = '{\"rationale\": \"test\", \"action\": {\"type\": \"click\", \"name\": \"button\"}}'
answer = {'type': 'click', 'name': 'button'}
prompt = [{'role': 'user', 'content': 'test'}]

for i, (func, weight) in enumerate(zip(env.rubric.funcs, env.rubric.weights)):
    try:
        r = func(completion, answer, prompt=prompt, info=answer)
        print(f'Reward {i}: {r:.3f} (weight={weight})')
    except Exception as e:
        print(f'Error in reward {i}: {e}')
"
```

## 🎯 Quick Test Only (If Limited Time)

If you just want to verify the code works:

```bash
# After Step 2 (setup), just run:
python tests/quick_test.py

# Should see:
# ✓ Environment loaded
# ✓ Rewards computed
# This proves the core implementation works!
```

---

**Total Time Estimate:**
- Setup: 5-7 minutes
- Quick test: 1 minute  
- SFT test: 2 minutes
- GRPO test: 3 minutes (if multi-GPU)
- Evaluation: 2 minutes

**Minimum Test (8 minutes):** Setup + Quick Test
**Full Test (15 minutes):** Everything