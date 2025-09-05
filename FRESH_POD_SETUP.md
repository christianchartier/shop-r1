# 🚀 FRESH POD SETUP - Complete Working Instructions

## Step 1: SSH into Fresh Pod
```bash
ssh -p 1234 root@[YOUR_POD_IP]

# ============================================
# ssh -p 1234 root@38.128.232.106
# ============================================
```

## Step 2: Run Complete Setup Script

**Option A: Download and Run Automated Script (Recommended)**
```bash
cd /workspace
wget https://raw.githubusercontent.com/christianchartier/shop-r1/main/RUNPOD_QUICK_SETUP.sh
chmod +x RUNPOD_QUICK_SETUP.sh
./RUNPOD_QUICK_SETUP.sh
```

**Option B: Manual Step-by-Step Installation**

```bash
# ============================================
# COMPLETE SETUP FOR FRESH POD - TESTED & WORKING
# ============================================

# 1. Navigate to workspace
cd /workspace || cd /ephemeral || cd ~
pwd

# 2. Install Python 3.11 (if not already installed)
echo "=== Installing Python 3.11 ==="
if command -v python3.11 &> /dev/null; then
    echo "✓ Python 3.11 already installed: $(python3.11 --version)"
else
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        python3.11 python3.11-venv python3.11-dev python3.11-distutils
    
    # Install pip for Python 3.11
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
fi

# 3. Clone repository
echo "=== Cloning Shop-R1 ==="
git clone https://github.com/christianchartier/shop-r1.git
cd shop-r1

# 4. Create Python 3.11 virtual environment
echo "=== Setting up Python Environment ==="
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate
python --version  # Should show Python 3.11.x

# 5. Install core dependencies
echo "=== Installing Dependencies ==="
python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.8
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118

# Install verifiers
python -m pip install 'verifiers @ git+https://github.com/willccbb/verifiers@main'

# Install ML libraries (note: transformers will auto-upgrade from verifiers dependencies)
python -m pip install \
    "accelerate>=0.30" \
    "peft>=0.11" \
    "datasets>=2.19" \
    requests \
    openai

# Install TRL separately (compatible with auto-upgraded transformers)
echo "Installing TRL..."
python -m pip install "trl>=0.11"

# 6. Install uv package manager (for vf-install)
echo "=== Installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 7. Setup Shop-R1 environment
echo "=== Setting up Shop-R1 Environment ==="
mkdir -p environments/shop_r1
cat > environments/shop_r1/pyproject.toml << 'EOF'
[project]
name = "shop-r1"
version = "0.1.0"
description = "Shop-R1 environment for verifiers"
requires-python = ">=3.8"
dependencies = [
    "verifiers",
    "requests>=2.31",
    "datasets>=2.19.0", 
    "transformers>=4.43",
    "accelerate>=0.30",
    "peft>=0.11",
]

[tool.setuptools]
py-modules = ["shop_r1", "synthesize"]

[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"
EOF

# 8. Install shop-r1 as editable package
python -m pip install -e .
vf-install shop-r1

# 9. Fix import issues
touch scripts/__init__.py

# 10. Apply FlashAttention2 fix
echo "=== Applying FlashAttention2 Fix ==="
python scripts/fix_flash_attention.py 2>/dev/null || python << 'PYTHON_FIX'
import re

# Read the SFT script
with open('scripts/sft_train.py', 'r') as f:
    content = f.read()

# Force SDPA attention in verifiers call
pattern1 = r'model, tokenizer = get_mat\(args\.model\)'
replacement1 = '''# Force SDPA attention
        import os
        os.environ['_TRANSFORMERS_ATTN_IMPLEMENTATION'] = 'sdpa'
        model_kwargs = {'attn_implementation': 'sdpa', 'torch_dtype': 'auto'}
        model, tokenizer = get_mat(args.model, use_liger=False, model_kwargs=model_kwargs)'''

if re.search(pattern1, content):
    content = re.sub(pattern1, replacement1, content)
    print("✓ Patched verifiers call to use SDPA")

# Also patch fallback transformers loading
pattern2 = r'model = AutoModelForCausalLM\.from_pretrained\([^)]+\)'
if 'attn_implementation' not in content and re.search(pattern2, content):
    def add_attn_impl(match):
        call = match.group(0)
        if 'attn_implementation' not in call:
            return call[:-1] + ',\n            attn_implementation="sdpa")'
        return call
    
    content = re.sub(pattern2, add_attn_impl, content)
    print("✓ Patched fallback AutoModelForCausalLM to use SDPA")

with open('scripts/sft_train.py', 'w') as f:
    f.write(content)

print("✓ FlashAttention2 fix applied successfully")
PYTHON_FIX

# 11. Create test dataset
echo "=== Creating Test Data ==="
mkdir -p data
cat > data/test.jsonl << 'EOF'
{"prompt": [{"role": "user", "content": "Search for laptop"}], "answer": {"type": "type_and_submit", "name": "search", "text": "laptop"}, "rationale": "Looking for a laptop"}
{"prompt": [{"role": "user", "content": "Click add to cart"}], "answer": {"type": "click", "name": "add_to_cart"}, "rationale": "Adding to cart"}
{"prompt": [{"role": "user", "content": "Done shopping"}], "answer": {"type": "terminate"}, "rationale": "Finished"}
EOF

echo "✓ Test dataset created"

# 12. Verify installation
echo "=== Verifying Installation ==="
python -c "
import torch, transformers, accelerate, verifiers
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print('✓ All core packages imported successfully')
"

# Show GPU info
nvidia-smi -L

echo "========================================="
echo "✅ SETUP COMPLETE!"
echo "========================================="

# IMPORTANT: Activate the virtual environment
cd /workspace/shop-r1
source .venv/bin/activate
```

## Step 3: Test SFT Training

```bash
# Quick test with minimal dataset
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

## Step 4: Test GRPO (Optional - Requires 2 GPUs)

**Note**: GRPO has known dataset iteration issues in the current environment. Skip to Step 5 (Evaluation) for more critical validation.

```bash
# Install vLLM and wandb for GRPO
python -m pip install "vllm==0.10.1.1" wandb

# Create larger test dataset
cat > data/test_large.jsonl << 'EOF'
{"prompt": [{"role": "user", "content": "Search for laptop"}], "answer": {"type": "type_and_submit", "name": "search", "text": "laptop"}, "rationale": "Looking for a laptop"}
{"prompt": [{"role": "user", "content": "Click add to cart"}], "answer": {"type": "click", "name": "add_to_cart"}, "rationale": "Adding to cart"}
{"prompt": [{"role": "user", "content": "Done shopping"}], "answer": {"type": "terminate"}, "rationale": "Finished"}
{"prompt": [{"role": "user", "content": "Search for headphones"}], "answer": {"type": "type_and_submit", "name": "search", "text": "headphones"}, "rationale": "Looking for headphones"}
{"prompt": [{"role": "user", "content": "Click product link"}], "answer": {"type": "click", "name": "product_link"}, "rationale": "Checking product"}
{"prompt": [{"role": "user", "content": "Add to wishlist"}], "answer": {"type": "click", "name": "wishlist"}, "rationale": "Saving for later"}
{"prompt": [{"role": "user", "content": "Search for books"}], "answer": {"type": "type_and_submit", "name": "search", "text": "books"}, "rationale": "Looking for books"}
{"prompt": [{"role": "user", "content": "End session"}], "answer": {"type": "terminate"}, "rationale": "Finished shopping"}
EOF

# Terminal 1: Start vLLM server (in tmux)
tmux new -s vllm
# Inside tmux:
cd /workspace/shop-r1 && source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python -m trl.scripts.vllm_serve \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.20
# Press Ctrl+B, then D to detach

# Terminal 2: Run GRPO (main terminal)
sleep 15
CUDA_VISIBLE_DEVICES=1 python scripts/rl_train_grpo.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset data/test_large.jsonl \
  --output_dir checkpoints/test_rl \
  --max_steps 2 \
  --save_steps 10 \
  --eval_steps 10 \
  --per_device_batch_size 1 \
  --num_generations 1 \
  --grad_accum 1 \
  --max_seq_len 1024 \
  --learning_rate 1e-7

# Clean up
tmux kill-session -t vllm
```

## Step 5: Run Evaluation (Most Important Test)

### Terminal 1: Start Evaluation vLLM Server
```bash
# Check GPU memory first
nvidia-smi

# Start tmux session
tmux new -s eval

# Inside tmux:
cd /workspace/shop-r1 && source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 --port 8001 \
  --dtype auto \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.70 \
  --disable-log-requests \
  --enforce-eager

# Press Ctrl+B, then D to detach
```

### Terminal 2: Run Evaluation
```bash
# Wait for server to load
sleep 30

# Set environment
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://localhost:8001/v1

# Test server connection
curl -s http://localhost:8001/v1/models | jq .

# Run evaluation (IMPORTANT: must specify -b and -k flags)
vf-eval shop-r1 -m Qwen/Qwen2.5-0.5B-Instruct -b http://localhost:8001/v1 -k EMPTY -n 5

# Clean up
tmux kill-session -t eval
```

## Troubleshooting

### If Python 3.11 installation hangs:
```bash
# Kill the hanging process and use the alternative method:
pkill -f apt
DEBIAN_FRONTEND=noninteractive apt-get install -y python3.11 python3.11-venv python3.11-dev
```

### If TRL installation conflicts with transformers:
```bash
# Install TRL with flexible version after transformers is settled
python -m pip install "trl>=0.11"  # Will find compatible version
```

### If vLLM server fails to start:
```bash
# Use reduced memory settings
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 --port 8001 \
  --dtype auto \
  --max-model-len 512 \
  --gpu-memory-utilization 0.50 \
  --disable-log-requests \
  --enforce-eager \
  --max-num-batched-tokens 512
```

## Known Issues

1. **GRPO Dataset Iterator**: GRPO training fails with "no single sample in epoch_iterator" error. This is a compatibility issue between verifiers GRPO trainer and the current environment. SFT training works correctly.

2. **FlashAttention2**: The setup automatically patches scripts to use SDPA attention instead of FlashAttention2, which is not available in the RunPod environment.

3. **TRL Version Conflict**: The verifiers package may install an older transformers version that conflicts with TRL. Install TRL after all other packages to resolve.

## Summary

✅ **Working Components:**
- Python 3.11 installation
- Shop-R1 repository setup
- SFT training pipeline
- vLLM server setup
- Evaluation framework

⚠️ **Known Limitations:**
- GRPO training (dataset iteration issue)
- FlashAttention2 (using SDPA workaround)

The setup is sufficient for testing Shop-R1's core functionality, with SFT training and evaluation being the most critical components for validating the implementation.