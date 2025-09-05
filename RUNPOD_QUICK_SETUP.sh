#!/bin/bash
# ============================================
# RUNPOD QUICK SETUP SCRIPT - Simplified & Reliable
# ============================================

echo "=== RunPod Shop-R1 Setup Script ==="
echo "Starting at: $(date)"

# 1. Navigate to workspace
cd /workspace || cd /ephemeral || cd ~ || exit 1
WORKSPACE=$(pwd)
echo "Working directory: $WORKSPACE"

# 2. Check if Python 3.11 is already installed
echo "=== Checking Python Installation ==="
if command -v python3.11 &> /dev/null; then
    echo "✓ Python 3.11 already installed: $(python3.11 --version)"
else
    echo "Installing Python 3.11..."
    
    # Use a simpler, more reliable installation approach
    apt-get update -qq
    
    # Install Python 3.11 directly without PPA (often faster on RunPod)
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        python3.11-distutils \
        2>&1 | grep -E "Setting up|Processing|Unpacking|Get:" || true
    
    # Verify installation
    if command -v python3.11 &> /dev/null; then
        echo "✓ Python 3.11 installed successfully: $(python3.11 --version)"
    else
        echo "❌ Python 3.11 installation failed. Trying alternative method..."
        
        # Alternative: Use deadsnakes PPA with timeout
        apt-get install -y software-properties-common
        add-apt-repository -y ppa:deadsnakes/ppa
        apt-get update -qq
        
        # Install with explicit timeout
        timeout 120 apt-get install -y python3.11 python3.11-venv python3.11-dev || {
            echo "❌ Python installation timed out. Please manually install Python 3.11"
            exit 1
        }
    fi
fi

# 3. Install pip for Python 3.11 if not present
if ! python3.11 -m pip --version &> /dev/null; then
    echo "Installing pip for Python 3.11..."
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
fi

# 4. Clone repository
echo "=== Cloning Shop-R1 Repository ==="
if [ -d "shop-r1" ]; then
    echo "Repository already exists. Pulling latest changes..."
    cd shop-r1
    git pull
else
    git clone https://github.com/christianchartier/shop-r1.git
    cd shop-r1
fi

# 5. Create and activate virtual environment
echo "=== Setting up Python Environment ==="
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate

# Verify we're using Python 3.11
PYTHON_VERSION=$(python --version 2>&1)
echo "Active Python: $PYTHON_VERSION"

if [[ ! "$PYTHON_VERSION" == *"3.11"* ]]; then
    echo "❌ Error: Not using Python 3.11 in venv"
    exit 1
fi

# 6. Install core dependencies
echo "=== Installing Dependencies ==="
python -m pip install --upgrade pip setuptools wheel

# Install PyTorch first (with CUDA 11.8)
echo "Installing PyTorch..."
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118

# Install verifiers
echo "Installing verifiers..."
python -m pip install 'verifiers @ git+https://github.com/willccbb/verifiers@main'

# Install core ML libraries (note: transformers will auto-upgrade from verifiers)
echo "Installing ML libraries..."
python -m pip install \
    "accelerate>=0.30" \
    "peft>=0.11" \
    "datasets>=2.19" \
    requests \
    openai

# 7. Install uv for vf-install
echo "=== Installing uv Package Manager ==="
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 8. Create pyproject.toml for shop-r1 environment
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

# 9. Install shop-r1 environment
echo "Installing shop-r1 environment..."
python -m pip install -e .

# Install TRL after shop-r1 setup (must be after transformers is finalized)
echo "Installing TRL (final step for ML libraries)..."
python -m pip install "trl>=0.11" || echo "Warning: TRL installation failed, you may need to install it manually"

# Run vf-install
echo "Running vf-install for shop-r1..."
vf-install shop-r1

# 10. Fix import issues
echo "=== Fixing Script Issues ==="
touch scripts/__init__.py

# 11. Apply FlashAttention2 and DataCollator fixes
echo "Applying SFT training fixes..."
python << 'PYTHON_FIX'
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
    # Find the AutoModelForCausalLM call and add attn_implementation
    def add_attn_impl(match):
        call = match.group(0)
        if 'attn_implementation' not in call:
            # Insert before the closing parenthesis
            return call[:-1] + ',\n            attn_implementation="sdpa")'
        return call
    
    content = re.sub(pattern2, add_attn_impl, content)
    print("✓ Patched fallback AutoModelForCausalLM to use SDPA")

# Fix data collator for tensor size mismatch
if 'DataCollatorForSeq2Seq' not in content:
    # Replace default_data_collator import with DataCollatorForSeq2Seq
    content = content.replace('default_data_collator,', 'DataCollatorForSeq2Seq,')
    content = content.replace('default_data_collator', 'DataCollatorForSeq2Seq')
    
    # Update the collator instantiation
    collator_pattern = r'collator = default_data_collator'
    if re.search(collator_pattern, content):
        content = re.sub(collator_pattern, 
                        'collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)', 
                        content)
    print("✓ Fixed data collator for tensor size mismatch")

# Write back the fixed content
with open('scripts/sft_train.py', 'w') as f:
    f.write(content)

print("✓ All SFT training fixes applied successfully")
PYTHON_FIX

# 12. Create test data
echo "=== Creating Test Data ==="
mkdir -p data
cat > data/test.jsonl << 'EOF'
{"prompt": [{"role": "user", "content": "Search for laptop"}], "answer": {"type": "type_and_submit", "name": "search", "text": "laptop"}, "rationale": "Looking for a laptop"}
{"prompt": [{"role": "user", "content": "Click add to cart"}], "answer": {"type": "click", "name": "add_to_cart"}, "rationale": "Adding to cart"}
{"prompt": [{"role": "user", "content": "Done shopping"}], "answer": {"type": "terminate"}, "rationale": "Finished"}
EOF

echo "✓ Test dataset created with $(wc -l < data/test.jsonl) examples"

# 13. Quick validation
echo "=== Quick Validation Test ==="
python -c "
import torch
import transformers
import accelerate
import verifiers

print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'Accelerate: {accelerate.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# Check if TRL is installed
try:
    import trl
    print(f'TRL: {trl.__version__}')
except ImportError:
    print('TRL: Not installed yet (install with: pip install trl>=0.11)')
    
print('✓ Core packages imported successfully')
"

# 14. Show GPU info
echo "=== GPU Information ==="
nvidia-smi -L 2>/dev/null || echo "No GPUs detected"

echo ""
echo "========================================="
echo "✅ SETUP COMPLETE!"
echo "========================================="
echo ""
echo "Environment ready at: $WORKSPACE/shop-r1"
echo ""
echo "⚠️  IMPORTANT: Activate the virtual environment:"
echo "   cd $WORKSPACE/shop-r1"
echo "   source .venv/bin/activate"
echo ""
echo "Next steps:"
echo "1. Test SFT training:"
echo "   python scripts/sft_train.py --dataset data/test.jsonl --model Qwen/Qwen2.5-0.5B-Instruct --output_dir checkpoints/test --epochs 1"
echo ""
echo "2. For GRPO testing (requires 2 GPUs and vLLM):"
echo "   python -m pip install vllm==0.10.1.1 wandb"
echo ""
echo "========================================="