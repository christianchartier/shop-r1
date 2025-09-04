#!/bin/bash
# Local testing script before deploying to Prime Intellect

set -e

echo "=== Shop-R1 Local Testing ==="

# 1. Test environment loading
echo "1. Testing environment import and loading..."
python3 -c "
from environments.shop_r1.shop_r1 import load_environment
env = load_environment(dataset_path=None)
print('✓ Environment loaded successfully')
print(f'  Parser: {env.parser.__class__.__name__}')
print(f'  Rubric functions: {len(env.rubric.funcs)}')
"

# 2. Test JSON parser
echo -e "\n2. Testing JSON parser..."
python3 -c "
from environments.shop_r1.shop_r1 import JSONActionParser
parser = JSONActionParser()

# Test valid JSON
valid = '{\"rationale\": \"Looking for a laptop\", \"action\": {\"type\": \"click\", \"name\": \"search_button\"}}'
parsed = parser.parse_answer(valid)
assert parsed is not None, 'Failed to parse valid JSON'
assert parsed['type'] == 'click', f\"Expected type=click, got {parsed.get('type')}\"
print('✓ Parser works for valid JSON')

# Test invalid JSON
invalid = 'not json'
parsed_invalid = parser.parse_answer(invalid)
assert parsed_invalid is None, 'Should return None for invalid JSON'
print('✓ Parser correctly rejects invalid JSON')
"

# 3. Test reward computation
echo -e "\n3. Testing reward computation..."
python3 -c "
from environments.shop_r1.shop_r1 import load_environment
env = load_environment(dataset_path=None, debug_rewards=False)

# Create test data
prompt = [{'role': 'user', 'content': 'Test prompt'}]
answer = {'type': 'click', 'name': 'add_to_cart'}
completion = '{\"rationale\": \"Adding item\", \"action\": {\"type\": \"click\", \"name\": \"add_to_cart\"}}'

# Compute rewards
rewards = []
for func, weight in zip(env.rubric.funcs, env.rubric.weights):
    r = func(completion, answer, prompt=prompt, info=answer)
    rewards.append(r * weight)
    
total = sum(rewards)
print(f'✓ Rewards computed: {total:.3f}')
print(f'  Components: {[f\"{r:.3f}\" for r in rewards]}')
"

# 4. Test dataset loading (if exists)
if [ -f "data/sft.jsonl" ]; then
    echo -e "\n4. Testing dataset loading..."
    python3 -c "
import json
with open('data/sft.jsonl', 'r') as f:
    lines = f.readlines()
    print(f'✓ Dataset has {len(lines)} examples')
    # Parse first line
    first = json.loads(lines[0])
    assert 'prompt' in first, 'Missing prompt field'
    assert 'answer' in first or 'action' in first, 'Missing answer/action field'
    print('✓ Dataset format is valid')
"
else
    echo -e "\n4. No dataset found at data/sft.jsonl - will need to generate"
fi

# 5. Test SFT script imports
echo -e "\n5. Testing SFT training script..."
python3 -c "
import sys
sys.argv = ['sft_train.py', '--help']
try:
    from scripts.sft_train import main
    print('✓ SFT script imports successfully')
except SystemExit:
    pass  # --help causes exit
"

# 6. Test RL script imports
echo -e "\n6. Testing RL training script..."
python3 -c "
import sys
sys.argv = ['rl_train_grpo.py', '--help']
try:
    from scripts.rl_train_grpo import main
    print('✓ RL script imports successfully')
except SystemExit:
    pass  # --help causes exit
"

echo -e "\n=== Basic tests passed! ==="
echo "Next steps:"
echo "1. Generate/prepare dataset: python environments/shop_r1/synthesize.py -o data/sft.jsonl -n 100"
echo "2. Test SFT locally (small run): python scripts/sft_train.py --dataset data/sft.jsonl --epochs 1 --save_steps 10"
echo "3. Deploy to Prime Intellect for full testing"