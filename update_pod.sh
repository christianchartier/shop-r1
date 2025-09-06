#!/bin/bash
# Script to update the pod with latest changes from GitHub

echo "=== Updating Shop-R1 on Pod ==="
echo ""

# Navigate to the shop-r1 directory
cd /workspace/shop-r1

# Check current branch and status
echo "Current git status:"
git status
echo ""

# Stash any local changes (in case there are uncommitted edits)
echo "Stashing any local changes..."
git stash
echo ""

# Fetch latest changes from GitHub
echo "Fetching latest from GitHub..."
git fetch origin main
echo ""

# Reset to match remote main branch exactly
echo "Resetting to match remote main..."
git reset --hard origin/main
echo ""

# Show the latest commit to confirm
echo "Latest commit:"
git log -1 --oneline
echo ""

# Verify the GRPO fix is present
echo "Verifying GRPO routing fix is in place..."
if grep -q "generation_client = AsyncOpenAI" scripts/rl_train_grpo.py; then
    echo "✅ GRPO routing fix found in scripts/rl_train_grpo.py"
else
    echo "❌ GRPO routing fix NOT found - may need to check the file"
fi
echo ""

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment is active: $VIRTUAL_ENV"
else
    echo "⚠️  Virtual environment not active. Run: source .venv/bin/activate"
fi
echo ""

echo "=== Update Complete ==="
echo ""
echo "Next steps:"
echo "1. Start TRL Communicator server on port 8000 (GPU 0)"
echo "2. Start OpenAI API server on port 8001 (GPU 1)"
echo "3. Run GRPO training with CUDA_VISIBLE_DEVICES=1"