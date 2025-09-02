import argparse
import json
from typing import Any, Dict, List

import verifiers as vf


ABLATIONS: Dict[str, Dict[str, Any]] = {
    # Full system (paper default weights)
    "full": {
        "w_format": 0.5,
        "w_rationale": 0.13,
        "w_type": 0.3,
        "w_click_attr_presence": 0.2,
        "w_type_submit_name_presence": 0.1,
        "w_type_submit_text_presence": 0.1,
        "w_type_submit_name_sim": 0.1,
        "enable_dars": True,
        "dars_factor": 1000.0,
    },
    # Remove SFT is a training-time condition (use base model) — here we just expose config parity.
    # Remove format reward
    "no_format": {
        "w_format": 0.0,
    },
    # Remove self-certainty
    "no_rationale": {
        "w_rationale": 0.0,
        "enable_self_certainty": False,
    },
    # Remove DARS
    "no_dars": {
        "enable_dars": False,
        "dars_factor": 0.0,
    },
    # Approximate "binary" by disabling sub-action rewards (keeps type only)
    "type_only": {
        "w_click_attr_presence": 0.0,
        "w_type_submit_name_presence": 0.0,
        "w_type_submit_text_presence": 0.0,
        "w_type_submit_name_sim": 0.0,
        "dars_factor": 0.0,
    },
}


def main():
    ap = argparse.ArgumentParser(description="Shop-R1 Ablations — run short GRPO for each config")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="ablation_results.json")
    ap.add_argument("--max_steps", type=int, default=100)
    ap.add_argument("--learning_rate", type=float, default=1e-7)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--per_device_batch_size", type=int, default=1)
    ap.add_argument("--num_generations", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    args = ap.parse_args()

    results: Dict[str, Any] = {}

    for name, cfg in ABLATIONS.items():
        env_args = dict(cfg)
        env_args.update({
            "dataset_path": args.dataset,
            "strict": False,
            "sim_threshold": 0.75,
        })
        env = vf.load_environment(env_id="shop-r1", **env_args)
        model, tok = vf.get_model_and_tokenizer(args.model)
        tr_args = vf.grpo_defaults(run_name=f"shop-r1-ablate-{name}")
        tr_args.max_steps = args.max_steps
        tr_args.learning_rate = args.learning_rate
        tr_args.temperature = args.temperature
        tr_args.per_device_train_batch_size = args.per_device_batch_size
        tr_args.num_generations = args.num_generations
        tr_args.gradient_accumulation_steps = args.grad_accum
        tr_args.beta = 0.001

        trainer = vf.GRPOTrainer(model=model, processing_class=tok, env=env, args=tr_args)
        trainer.train()
        # Collect last logged reward mean if available
        try:
            r = getattr(trainer, "last_eval_reward", None)
        except Exception:
            r = None
        results[name] = {"reward": r}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

