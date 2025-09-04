# Shop-R1 Testing Guide

## 🚀 Quick Start on Prime Intellect Pod

```bash
# SSH into your pod
ssh -p 1234 root@62.169.159.61

# Clone and setup
cd /ephemeral
git clone https://github.com/christianchartier/shop-r1.git
cd shop-r1
python3.11 -m venv .venv311
source .venv311/bin/activate

# Install dependencies
pip install -U pip
pip install -e .
pip install 'verifiers @ git+https://github.com/willccbb/verifiers@main'
vf-install shop-r1

# Run quick test
python tests/quick_test.py
```

## 📁 Repository Structure

```
shop-r1/
├── environments/         # Main implementation
│   └── shop_r1/
│       ├── __init__.py
│       ├── shop_r1.py   # Core environment with rewards
│       └── synthesize.py # Data generation
├── scripts/             # Training scripts
│   ├── sft_train.py     # Supervised fine-tuning
│   └── rl_train_grpo.py # GRPO reinforcement learning
├── tests/               # All testing code
│   ├── quick_test.py    # Quick validation
│   └── README.md        # Testing documentation
├── docs/                # Documentation
│   └── internal/        # Internal docs and checklists
└── data/                # Dataset storage (create as needed)
```

## 🧪 Testing Workflow

### 1. Environment Validation
```bash
python tests/quick_test.py
```
✅ Should see: Environment loads, rewards computed, parser works

### 2. SFT Training Test
```bash
# Create minimal dataset
cat > data/test.jsonl << 'EOF'
{"prompt": [{"role": "user", "content": "Search"}], "answer": {"type": "click", "name": "search"}}
EOF

# Run SFT (1 epoch, tiny model)
python scripts/sft_train.py \
  --dataset data/test.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output_dir checkpoints/test_sft \
  --epochs 1 \
  --save_steps 1
```

### 3. GRPO Test (if 2+ GPUs)
```bash
# Terminal 1: Start vLLM server
CUDA_VISIBLE_DEVICES=0 python -m trl.scripts.vllm_serve \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 --port 8000

# Terminal 2: Run GRPO
CUDA_VISIBLE_DEVICES=1 python scripts/rl_train_grpo.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset data/test.jsonl \
  --output_dir checkpoints/test_rl \
  --max_steps 2
```

### 4. Evaluation
```bash
vf-eval shop-r1 -m Qwen/Qwen2.5-0.5B-Instruct -s -n 5
```

## ⚠️ Common Issues & Fixes

### Python Version
- **Need**: Python 3.9+ for verifiers
- **Fix**: Use `python3.11` on Prime Intellect pods

### GPU Memory
- **Issue**: OOM errors
- **Fix**: Reduce batch size or use smaller model

### GRPO Connection
- **Issue**: `client_device_uuid` error
- **Fix**: Already patched in `rl_train_grpo.py`

### Import Errors
- **Issue**: `ModuleNotFoundError: verifiers`
- **Fix**: Run `vf-install shop-r1` after pip install

## 📊 Expected Results

### Quick Test
```
✓ Environment loaded
✓ Parser works
✓ Rewards computed
  Total reward: ~0.8-1.0 for correct action
```

### SFT Training
- Loss decreases from ~3.0 to <2.0
- No NaN values
- Saves checkpoints

### Evaluation
- Format reward: >0.8
- Action accuracy: >0.3
- Overall reward: >0.5

## 📝 For Full Testing

See `docs/internal/pod_commands.txt` for complete step-by-step commands.