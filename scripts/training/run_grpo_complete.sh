#!/bin/bash
# Complete GRPO Training Setup and Execution Script
# This script handles the entire dual-server setup and training process

set -e  # Exit on error

echo "==================================================="
echo "     Shop-R1 GRPO Training - Complete Setup"
echo "==================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TRL_PORT=8000
OPENAI_PORT=8001
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
DATASET="data/rl.jsonl"
OUTPUT_DIR="checkpoints/rl_shop_r1_$(date +%Y%m%d_%H%M%S)"

# Function to check if a port is in use
check_port() {
    local port=$1
    if netstat -tuln 2>/dev/null | grep -q ":$port "; then
        return 0
    else
        return 1
    fi
}

# Step 1: Environment Setup
echo -e "${BLUE}Step 1: Setting up environment...${NC}"
cd /workspace/shop-r1
source .venv/bin/activate

# Disable Weights & Biases login unless explicitly enabled
export WANDB_DISABLED=${WANDB_DISABLED:-true}

# Check for vLLM installation
if ! python -c "import vllm" 2>/dev/null; then
    echo "Installing vLLM..."
    pip install "vllm==0.10.1.1" || pip install vllm
fi

# Ensure wandb is installed regardless of vLLM presence
if ! python -c "import wandb" 2>/dev/null; then
    echo "Installing wandb..."
    pip install wandb
fi

# Step 2: Dataset Creation
echo -e "${BLUE}Step 2: Preparing dataset...${NC}"
if [ ! -f "$DATASET" ]; then
    echo "Creating RL dataset..."
    python environments/shop_r1/synthesize.py -o "$DATASET" -n 200 --seed 7
else
    echo "Dataset already exists: $DATASET"
fi

# Step 3: Server Management
echo -e "${BLUE}Step 3: Starting dual vLLM servers...${NC}"

# Clean up any existing servers
echo "Cleaning up existing servers..."
tmux kill-session -t vllm_trl 2>/dev/null || true
tmux kill-session -t vllm_oai 2>/dev/null || true
sleep 2

# Start TRL Communicator Server
echo -e "${YELLOW}Starting TRL Communicator Server (GPU 0, port $TRL_PORT)...${NC}"
tmux new -d -s vllm_trl "cd /workspace/shop-r1 && source .venv/bin/activate && \
  CUDA_VISIBLE_DEVICES=0 python -m trl.scripts.vllm_serve \
  --model $MODEL \
  --host 0.0.0.0 --port $TRL_PORT \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.60 2>&1 | tee /tmp/vllm_trl.log"

# Wait for TRL server
echo "Waiting for TRL server to start..."
for i in {1..30}; do
    if curl -s http://localhost:$TRL_PORT/get_world_size/ >/dev/null 2>&1; then
        echo -e "${GREEN}✅ TRL Communicator Server is running${NC}"
        break
    fi
    echo -n "."
    sleep 2
done

if ! curl -s http://localhost:$TRL_PORT/get_world_size/ >/dev/null 2>&1; then
    echo -e "${RED}❌ Failed to start TRL server. Check logs:${NC}"
    echo "tmux attach -t vllm_trl"
    exit 1
fi

# Start OpenAI API Server
echo -e "${YELLOW}Starting OpenAI API Server (GPU 1, port $OPENAI_PORT)...${NC}"
tmux new -d -s vllm_oai "cd /workspace/shop-r1 && source .venv/bin/activate && \
  CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model $MODEL \
  --host 0.0.0.0 --port $OPENAI_PORT \
  --dtype auto \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.50 \
  --disable-log-requests \
  --enforce-eager \
  --max-num-batched-tokens 512 2>&1 | tee /tmp/vllm_oai.log"

# Wait for OpenAI server
echo "Waiting for OpenAI server to start..."
for i in {1..40}; do
    if curl -s http://localhost:$OPENAI_PORT/v1/models >/dev/null 2>&1; then
        echo -e "${GREEN}✅ OpenAI API Server is running${NC}"
        break
    fi
    echo -n "."
    sleep 2
done

if ! curl -s http://localhost:$OPENAI_PORT/v1/models >/dev/null 2>&1; then
    echo -e "${RED}❌ Failed to start OpenAI server. Check logs:${NC}"
    echo "tmux attach -t vllm_oai"
    exit 1
fi

# Step 4: Verify Server Setup
echo -e "${BLUE}Step 4: Verifying server setup...${NC}"
echo "TRL Communicator endpoints:"
curl -s http://localhost:$TRL_PORT/get_world_size/ | python -m json.tool || echo "Failed to get world size"

echo ""
echo "OpenAI API models:"
curl -s http://localhost:$OPENAI_PORT/v1/models | python -m json.tool | head -20 || echo "Failed to get models"

# Step 5: Run GRPO Training
echo ""
echo -e "${BLUE}Step 5: Running GRPO training...${NC}"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Set environment variables
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://localhost:$OPENAI_PORT/v1

# Training parameters
if [ "$1" == "--quick" ]; then
    echo -e "${YELLOW}Running quick test (1 step)...${NC}"
    MAX_STEPS=1
    GRAD_ACCUM=1
else
    echo -e "${YELLOW}Running full training (50 steps)...${NC}"
    MAX_STEPS=50
    GRAD_ACCUM=8
fi

# Run training
CUDA_VISIBLE_DEVICES=1 python scripts/rl_train_grpo.py \
  --model $MODEL \
  --dataset $DATASET \
  --output_dir $OUTPUT_DIR \
  --strict \
  --sim_threshold 0.75 \
  --alpha 0.005 \
  --beta 0.001 \
  --dars_factor 1000 \
  --temperature 0.6 \
  --per_device_batch_size 1 \
  --num_generations 1 \
  --grad_accum $GRAD_ACCUM \
  --max_steps $MAX_STEPS \
  --save_steps 25 \
  --eval_steps 25 \
  --max_seq_len 1024 \
  --learning_rate 1e-7 2>&1 | tee /tmp/grpo_training.log

# Check training result
if grep -q "train_runtime" /tmp/grpo_training.log; then
    echo ""
    echo -e "${GREEN}==================================================="
    echo "     ✅ GRPO Training Completed Successfully!"
    echo "===================================================${NC}"
    echo ""
    echo "Training metrics:"
    grep -E "loss|rewards|train_runtime" /tmp/grpo_training.log | tail -5
    echo ""
    echo "Output saved to: $OUTPUT_DIR"
else
    echo ""
    echo -e "${RED}==================================================="
    echo "     ❌ GRPO Training Failed"
    echo "===================================================${NC}"
    echo "Check logs at: /tmp/grpo_training.log"
fi

# Step 6: Cleanup Options
echo ""
echo -e "${BLUE}Server Management:${NC}"
echo "  Monitor TRL server:    tmux attach -t vllm_trl    (Ctrl+B,D to detach)"
echo "  Monitor OpenAI server: tmux attach -t vllm_oai    (Ctrl+B,D to detach)"
echo ""
echo "  Stop servers:          tmux kill-session -t vllm_trl && tmux kill-session -t vllm_oai"
echo ""
echo "  View logs:             tail -f /tmp/vllm_trl.log"
echo "                         tail -f /tmp/vllm_oai.log"
echo ""

# Ask if user wants to keep servers running
read -p "Keep servers running for more experiments? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Stopping servers..."
    tmux kill-session -t vllm_trl 2>/dev/null || true
    tmux kill-session -t vllm_oai 2>/dev/null || true
    echo -e "${GREEN}Servers stopped.${NC}"
else
    echo -e "${GREEN}Servers kept running. Use commands above to manage them.${NC}"
fi
