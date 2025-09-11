.PHONY: setup test-shop-r1 eval-tiny runpod grpo-quick grpo-50 multiturn-smoke

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

# Run GRPO quick smoke (1 step) with dual vLLM servers
grpo-quick:
	chmod +x scripts/training/run_grpo_complete.sh
	./scripts/training/run_grpo_complete.sh --quick

# Run GRPO short training (50 steps) with dual vLLM servers
grpo-50:
	chmod +x scripts/training/run_grpo_complete.sh
	./scripts/training/run_grpo_complete.sh

# Tiny multi-turn smoke: build one 2-step episode and construct env
multiturn-smoke:
	@mkdir -p data
	@printf '{"steps":[{"prompt":[{"role":"user","content":"context t1"}],"answer":{"type":"type_and_submit","name":"search_input","text":"laptop"}},{"prompt":[{"role":"user","content":"context t2"}],"answer":{"type":"click","name":"view_details"}}]}' > data/episodes.jsonl
	uv run python -c "from environments.shop_r1.shop_r1 import load_multiturn_environment as load; env=load(dataset_path='data/episodes.jsonl', max_episodes=1, strict=True); ds=getattr(env,'dataset',[]); funcs=getattr(getattr(env,'rubric',None),'funcs',[]); print('episodes:', len(ds)); print('rubric_funcs:', len(funcs)); print('multiturn smoke: OK')" 
