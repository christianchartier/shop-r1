.PHONY: setup test-shop-r1 eval-tiny runpod

setup:
	uv pip install -e .
	uv pip install pytest

test-shop_r1:
	uv run pytest -q tests/test_shop_r1_smoke.py

eval-tiny:
	uv run python environments/shop_r1/synthesize.py -o data/shop_r1_tiny.jsonl -n 20 --seed 42
	uv run python scripts/eval_paper_metrics.py --dataset data/shop_r1_tiny.jsonl --max_examples 20 --output results/shop_r1_results.json

runpod:
	@echo "See docs/runpod.md for full one-shot RunPod instructions."
