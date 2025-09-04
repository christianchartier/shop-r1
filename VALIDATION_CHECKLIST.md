# Shop-R1 Implementation Validation Checklist

## Pre-Submission Checklist (Based on Prime Intellect Standards)

### ✅ Code Quality
- [x] Run `ruff check --fix .` on all code
- [x] Fixed import errors  
- [ ] Verify no "vibe-coded slop" - implementation is faithful to paper
- [x] Type hints added where appropriate
- [x] Error handling with fallbacks

### ✅ Implementation Correctness
- [x] **Reward Structure** matches Table 1 from paper:
  - Format reward: 0.5
  - Rationale reward: 0.13 (self-certainty)
  - Action type reward: 0.3
  - Sub-action rewards correctly weighted
- [x] **DARS** (Difficulty-Aware Reward Scaling) implemented
- [x] **Self-certainty** reward via KL divergence approximation
- [x] **Hierarchical rewards** prevent reward hacking

### ⚠️ Testing Requirements
- [ ] **Small-scale eval with Qwen3-30B-A3B or Qwen3-4B**
- [ ] **Run with `vf-eval -s`** and include outputs
- [ ] **Reward scores pass smell test**
- [ ] **SFT training converges**
- [ ] **GRPO training runs without errors**

### 📦 Package Configuration
- [ ] Update to `verifiers>=0.1.3`
- [x] Include proper tags in `pyproject.toml`
- [x] Add description and dependencies
- [ ] Link to source implementation

### 📝 Documentation
- [ ] Clear PR note explaining implementation
- [ ] README with:
  - [ ] Link to paper
  - [ ] Installation instructions  
  - [ ] Usage examples
  - [ ] Expected results
- [ ] Include author/social links for credit

## Known Issues to Fix

### 1. **Multi-GPU GRPO Communication**
- Issue: `client_device_uuid` missing in init_communicator
- Fix: Already patched in `rl_train_grpo.py` lines 108-170

### 2. **Python Version Compatibility**
- Issue: Local Python 3.8 too old for verifiers
- Fix: Use Python 3.11 on Prime Intellect pods

### 3. **Missing Test Data**
- Issue: No real shopping session data
- Fix: Use synthetic data generator or small test set

## Test Commands Sequence

```bash
# On Prime Intellect Pod (Python 3.11+)

# 1. Install and setup
pip install -e .
vf-install shop-r1

# 2. Generate test data
python environments/shop_r1/synthesize.py -o data/test.jsonl -n 100

# 3. Quick SFT test (should complete in <5 min)
python scripts/sft_train.py \
  --dataset data/test.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output_dir checkpoints/sft_test \
  --epochs 1 \
  --save_steps 10

# 4. Evaluate with verifiers
vf-eval shop-r1 -m Qwen/Qwen2.5-0.5B-Instruct -s -n 10

# 5. If multi-GPU available, test GRPO
# Terminal 1:
CUDA_VISIBLE_DEVICES=0 python -m trl.scripts.vllm_serve \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 --port 8000

# Terminal 2:
CUDA_VISIBLE_DEVICES=1 python scripts/rl_train_grpo.py \
  --model checkpoints/sft_test \
  --dataset data/test.jsonl \
  --output_dir checkpoints/rl_test \
  --max_steps 10
```

## Expected Results

### SFT Training
- Loss should decrease from ~3.0 to <2.0
- No NaN or inf values
- Checkpoints saved successfully

### Evaluation (vf-eval)
- Format reward: ~0.8-1.0 (most outputs should be valid JSON)
- Action type accuracy: >0.3 (better than random)
- Overall reward: >0.5

### GRPO Training  
- Successful connection to vLLM server
- Rewards computed without errors
- Model updates without NaN

## Submission Requirements Met?

- [ ] Code runs without errors
- [ ] Evaluation scores reasonable
- [ ] Documentation complete
- [ ] Following paper specifications
- [ ] Multi-GPU support (if applicable)

## Next Steps

1. **Test on Prime Intellect pod** with Python 3.11+
2. **Run evaluation** with proper model (Qwen3-30B-A3B)
3. **Document results** in README
4. **Submit PR** with all requirements met