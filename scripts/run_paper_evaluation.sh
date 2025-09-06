#!/bin/bash
# Run complete evaluation matching the paper's Table 2
# This script evaluates different model configurations and generates results

set -e

echo "=============================================="
echo "    Shop-R1 Paper Evaluation (Table 2)"
echo "=============================================="
echo ""

# Configuration
DATASET="${1:-data/test.jsonl}"
OUTPUT_DIR="${2:-results/table2}"
MODEL_BASE="Qwen/Qwen2.5-3B-Instruct"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to run evaluation
run_eval() {
    local model_name=$1
    local checkpoint=$2
    local output_file=$3
    
    echo "Evaluating: $model_name"
    echo "-------------------------------------------"
    
    if [ -z "$checkpoint" ]; then
        # Use base model (zero-shot)
        python scripts/eval_paper_metrics.py \
            --dataset "$DATASET" \
            --model_alias "local-qwen" \
            --temperature 0.0 \
            --output "$output_file" \
            --quiet
    else
        # Use checkpoint
        python scripts/eval_paper_metrics.py \
            --dataset "$DATASET" \
            --use_checkpoint "$checkpoint" \
            --temperature 0.0 \
            --output "$output_file" \
            --quiet
    fi
    
    # Extract metrics for table
    if [ -f "$output_file" ]; then
        exact=$(python -c "import json; print(f\"{json.load(open('$output_file'))['exact_action_acc']:.2%}\")")
        type_acc=$(python -c "import json; print(f\"{json.load(open('$output_file'))['action_type_acc']:.2%}\")")
        f1=$(python -c "import json; print(f\"{json.load(open('$output_file'))['action_type_f1']:.2%}\")")
        
        echo "$model_name: Exact=$exact, Type=$type_acc, F1=$f1"
        echo ""
    fi
}

# Header for results table
echo "Generating Table 2 Results..."
echo ""
echo "Model                          Exact    Type     F1"
echo "------------------------------------------------------"

# 1. Zero-shot prompting (baseline)
echo -n "Zero-shot prompting            "
run_eval "Zero-shot" "" "$OUTPUT_DIR/zero_shot.json" | tail -1

# 2. SFT only (if checkpoint exists)
if [ -d "checkpoints/sft_shop_r1" ]; then
    echo -n "SFT                            "
    run_eval "SFT" "checkpoints/sft_shop_r1" "$OUTPUT_DIR/sft.json" | tail -1
fi

# 3. SFT + RL Binary (if checkpoint exists)
if [ -d "checkpoints/rl_binary" ]; then
    echo -n "SFT + RL (Binary)              "
    run_eval "SFT+RL-Binary" "checkpoints/rl_binary" "$OUTPUT_DIR/sft_rl_binary.json" | tail -1
fi

# 4. Shop-R1 (Our method)
if [ -d "checkpoints/rl_shop_r1" ]; then
    echo -n "Shop-R1 (Ours)                 "
    run_eval "Shop-R1" "checkpoints/rl_shop_r1" "$OUTPUT_DIR/shop_r1.json" | tail -1
fi

echo "------------------------------------------------------"
echo ""

# Generate consolidated results JSON
echo "Generating consolidated results..."
python - <<EOF
import json
import os
from pathlib import Path

output_dir = Path("$OUTPUT_DIR")
results = {}

# Load all result files
for result_file in output_dir.glob("*.json"):
    name = result_file.stem
    with open(result_file) as f:
        data = json.load(f)
        results[name] = {
            "exact_action_acc": data["exact_action_acc"],
            "action_type_acc": data["action_type_acc"],
            "action_type_f1": data["action_type_f1"],
            "total_samples": data["total_samples"]
        }

# Save consolidated results
with open(output_dir / "consolidated_table2.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Consolidated results saved to: {output_dir}/consolidated_table2.json")
EOF

echo ""
echo "=============================================="
echo "Evaluation Complete!"
echo "Results saved in: $OUTPUT_DIR"
echo "=============================================="