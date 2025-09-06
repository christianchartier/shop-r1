#!/bin/bash
# Complete test script for GRPO with dual-server setup

echo "=== GRPO Dual-Server Test Script ==="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if a port is listening
check_port() {
    local port=$1
    if netstat -tuln | grep -q ":$port "; then
        return 0
    else
        return 1
    fi
}

# Kill any existing servers
echo "Cleaning up any existing servers..."
tmux kill-session -t vllm_trl 2>/dev/null || true
tmux kill-session -t vllm_oai 2>/dev/null || true
pkill -f "trl.scripts.vllm_serve" 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 3

# Navigate to shop-r1 and activate environment
cd /workspace/shop-r1
source .venv/bin/activate

# Start TRL Communicator Server (GPU 0, Port 8000)
echo -e "${YELLOW}Starting TRL Communicator Server on port 8000 (GPU 0)...${NC}"
tmux new -d -s vllm_trl "cd /workspace/shop-r1 && source .venv/bin/activate && \
  CUDA_VISIBLE_DEVICES=0 python -m trl.scripts.vllm_serve \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.60"

# Wait for TRL server to start
echo "Waiting for TRL server to start..."
for i in {1..30}; do
    if curl -s http://localhost:8000/get_world_size/ >/dev/null 2>&1; then
        echo -e "${GREEN}✅ TRL Communicator Server is running on port 8000${NC}"
        break
    fi
    echo -n "."
    sleep 2
done

# Verify TRL server
if ! curl -s http://localhost:8000/get_world_size/ >/dev/null 2>&1; then
    echo -e "${RED}❌ TRL Communicator Server failed to start on port 8000${NC}"
    echo "Check logs with: tmux attach -t vllm_trl"
    exit 1
fi

# Start OpenAI API Server (GPU 1, Port 8001)
echo -e "${YELLOW}Starting OpenAI API Server on port 8001 (GPU 1)...${NC}"
tmux new -d -s vllm_oai "cd /workspace/shop-r1 && source .venv/bin/activate && \
  CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 --port 8001 \
  --dtype auto \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.50 \
  --disable-log-requests \
  --enforce-eager \
  --max-num-batched-tokens 512"

# Wait for OpenAI server to start
echo "Waiting for OpenAI server to start..."
for i in {1..30}; do
    if curl -s http://localhost:8001/v1/models >/dev/null 2>&1; then
        echo -e "${GREEN}✅ OpenAI API Server is running on port 8001${NC}"
        break
    fi
    echo -n "."
    sleep 2
done

# Verify OpenAI server
if ! curl -s http://localhost:8001/v1/models >/dev/null 2>&1; then
    echo -e "${RED}❌ OpenAI API Server failed to start on port 8001${NC}"
    echo "Check logs with: tmux attach -t vllm_oai"
    exit 1
fi

echo ""
echo -e "${GREEN}=== Both servers are running! ===${NC}"
echo ""

# Test the servers
echo "Testing server endpoints:"
echo -n "  TRL /get_world_size: "
if curl -s http://localhost:8000/get_world_size/ | grep -q "world_size"; then
    echo -e "${GREEN}✅${NC}"
else
    echo -e "${RED}❌${NC}"
fi

echo -n "  OpenAI /v1/models: "
if curl -s http://localhost:8001/v1/models | grep -q "Qwen"; then
    echo -e "${GREEN}✅${NC}"
else
    echo -e "${RED}❌${NC}"
fi

echo ""
echo "=== Running GRPO Training Test ==="
echo ""

# Set environment variables
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://localhost:8001/v1

# Create a minimal test dataset if it doesn't exist
if [ ! -f data/rl.jsonl ]; then
    echo "Creating test dataset..."
    python environments/shop_r1/synthesize.py -o data/rl.jsonl -n 10 --seed 7
fi

# Run GRPO training with minimal settings
echo "Starting GRPO training (minimal test)..."
CUDA_VISIBLE_DEVICES=1 timeout 60 python scripts/rl_train_grpo.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset data/rl.jsonl \
  --output_dir /tmp/test_grpo_$(date +%s) \
  --strict \
  --sim_threshold 0.75 \
  --alpha 0.005 \
  --beta 0.001 \
  --dars_factor 1000 \
  --temperature 0.6 \
  --per_device_batch_size 1 \
  --num_generations 1 \
  --grad_accum 1 \
  --max_steps 1 \
  --save_steps 25 \
  --eval_steps 25 \
  --max_seq_len 1024 \
  --learning_rate 1e-7 2>&1 | tee /tmp/grpo_test.log

echo ""
echo "=== Checking for routing issues ==="
echo ""

# Check TRL server logs for 404s
echo "Checking TRL server logs for 404 errors..."
if tmux capture-pane -pt vllm_trl -S -30 | grep -q "404 Not Found"; then
    echo -e "${RED}❌ Found 404 errors in TRL server - routing issue detected!${NC}"
    echo "Recent TRL server logs:"
    tmux capture-pane -pt vllm_trl -S -10
else
    echo -e "${GREEN}✅ No 404 errors in TRL server${NC}"
fi

# Check OpenAI server logs for successful requests
echo ""
echo "Checking OpenAI server logs for chat completions..."
if tmux capture-pane -pt vllm_oai -S -30 | grep -q "POST /v1/chat/completions"; then
    echo -e "${GREEN}✅ OpenAI server received chat completion requests${NC}"
else
    echo -e "${YELLOW}⚠️  No chat completion requests seen in OpenAI server${NC}"
fi

echo ""
echo "=== Test Complete ==="
echo ""
echo "To monitor servers:"
echo "  TRL logs: tmux attach -t vllm_trl"
echo "  OpenAI logs: tmux attach -t vllm_oai"
echo ""
echo "To clean up:"
echo "  tmux kill-session -t vllm_trl"
echo "  tmux kill-session -t vllm_oai"