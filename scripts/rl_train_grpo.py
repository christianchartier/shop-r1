import argparse
import os

import verifiers as vf


def main():
    ap = argparse.ArgumentParser(description="Shop-R1 GRPO training (SFT + RL)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="HF model id or SFT checkpoint path")
    ap.add_argument("--dataset", required=True, help="JSONL path for RL (prompt+answer per step)")
    ap.add_argument("--output_dir", default="checkpoints/rl_shop_r1", help="Output directory")
    # Environment args (paper defaults)
    ap.add_argument("--strict", action="store_true", help="Strict schema enforcement during RL rollouts")
    ap.add_argument("--sim_threshold", type=float, default=0.75)
    ap.add_argument("--alpha", type=float, default=0.13, help="Rationale weight (w_rationale)")
    ap.add_argument("--beta", type=float, default=0.001, help="KL penalty coefficient")
    ap.add_argument("--dars_factor", type=float, default=1000.0)
    # Training args (paper-like)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--learning_rate", type=float, default=1e-7)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--per_device_batch_size", type=int, default=1)
    ap.add_argument("--num_generations", type=int, default=8, help="Completions per prompt per step")
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_seq_len", type=int, default=32768)
    ap.add_argument("--eval_steps", type=int, default=100)
    ap.add_argument("--save_steps", type=int, default=100)
    args = ap.parse_args()

    # Environment with Shop-R1 rewards
    env = vf.load_environment(
        env_id="shop-r1",
        dataset_path=args.dataset,
        strict=args.strict,
        sim_threshold=args.sim_threshold,
        # Reward weights per paper figure
        w_format=0.5,
        w_rationale=args.alpha,
        w_type=0.3,
        w_click_attr_presence=0.2,
        w_type_submit_name_presence=0.1,
        w_type_submit_text_presence=0.1,
        w_type_submit_name_sim=0.1,
        # DARS controls
        enable_dars=True,
        dars_factor=args.dars_factor,
    )

    # Base model (optionally SFT checkpoint)
    model, tokenizer = vf.get_model_and_tokenizer(args.model)

    # GRPO Trainer hyperparameters (vLLM server must be running)
    tr_args = vf.grpo_defaults(run_name="shop-r1-grpo")
    tr_args.output_dir = args.output_dir
    tr_args.learning_rate = args.learning_rate
    tr_args.max_steps = args.max_steps
    tr_args.per_device_train_batch_size = args.per_device_batch_size
    tr_args.num_generations = args.num_generations
    tr_args.gradient_accumulation_steps = args.grad_accum
    tr_args.max_seq_len = args.max_seq_len
    tr_args.temperature = args.temperature
    tr_args.beta = args.beta
    tr_args.eval_strategy = "steps"
    tr_args.eval_steps = args.eval_steps
    tr_args.save_strategy = "steps"
    tr_args.save_steps = args.save_steps

    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=env,
        args=tr_args,
    )
    trainer.train()


if __name__ == "__main__":
    main()

