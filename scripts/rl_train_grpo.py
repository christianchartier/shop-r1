import argparse
import os

import verifiers as vf
import uuid
from openai import AsyncOpenAI


def main():
    ap = argparse.ArgumentParser(description="Shop-R1 GRPO training (SFT + RL)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="HF model id or SFT checkpoint path")
    ap.add_argument("--dataset", required=True, help="JSONL path for RL (prompt+answer per step)")
    ap.add_argument("--output_dir", default="checkpoints/rl_shop_r1", help="Output directory")
    # Environment args (paper defaults)
    ap.add_argument("--strict", action="store_true", help="Strict schema enforcement during RL rollouts")
    ap.add_argument("--sim_threshold", type=float, default=0.75)
    # Paper fidelity: alpha ≈ 0.005 for rationale term in RL objective
    ap.add_argument("--alpha", type=float, default=0.005, help="Rationale weight (w_rationale)")
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

    # Environment with Shop-R1 rewards (fallback to direct import if verifiers API is missing)
    load_env_fn = getattr(vf, "load_environment", None)
    env_kwargs = dict(
        dataset_path=args.dataset,
        strict=args.strict,
        sim_threshold=args.sim_threshold,
        w_format=0.5,
        w_rationale=args.alpha,
        w_type=0.3,
        w_click_attr_presence=0.2,
        w_type_submit_name_presence=0.1,
        w_type_submit_text_presence=0.1,
        w_type_submit_name_sim=0.1,
        enable_dars=True,
        dars_factor=args.dars_factor,
    )
    if callable(load_env_fn):
        env = load_env_fn(env_id="shop-r1", **env_kwargs)
    else:
        from environments.shop_r1.shop_r1 import load_environment as load_env
        env = load_env(**env_kwargs)

    # Base model (optionally SFT checkpoint)
    # Always load via Transformers to avoid accelerate offload hooks in some verifiers builds
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        attn_implementation="sdpa",
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    )
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # GRPO Trainer hyperparameters (vLLM server must be running)
    tr_args = vf.grpo_defaults(run_name="shop-r1-grpo")
    tr_args.output_dir = args.output_dir
    # Avoid W&B by default; use report_to="none" to prevent integration
    try:
        tr_args.report_to = "none"  # type: ignore[attr-defined]
    except Exception:
        pass
    tr_args.learning_rate = args.learning_rate
    tr_args.max_steps = args.max_steps
    tr_args.per_device_train_batch_size = args.per_device_batch_size
    tr_args.num_generations = args.num_generations
    tr_args.gradient_accumulation_steps = args.grad_accum
    tr_args.max_seq_len = args.max_seq_len
    tr_args.temperature = args.temperature
    # Ensure logprobs/top_logprobs are requested for self-certainty; enforce JSON object shape
    # Set max_tokens to a safe value that leaves room for input tokens
    try:
        tr_args.sampling_args = {
            "temperature": args.temperature,
            "max_tokens": 160,  # Conservative value for responses
            "logprobs": True,
            "top_logprobs": 5,
            "response_format": {"type": "json_object"},
        }
        # Override max_tokens in trainer args to prevent it from being set to max_seq_len
        tr_args.max_tokens = 160  # This prevents the trainer from using max_seq_len
    except Exception:
        pass
    tr_args.beta = args.beta
    # Avoid moving a model that may have accelerate/deepspeed hooks; let the launcher place it
    try:
        tr_args.place_model_on_device = False  # type: ignore[attr-defined]
    except Exception:
        pass
    # As an additional safeguard, monkey‑patch HF Trainer's device move to a no‑op
    try:
        from transformers.trainer import Trainer as HFTrainer  # type: ignore

        def _noop_move_model_to_device(self, model, device):  # type: ignore[override]
            return model

        HFTrainer._move_model_to_device = _noop_move_model_to_device  # type: ignore[attr-defined]
    except Exception:
        pass
    tr_args.eval_strategy = "steps"
    tr_args.eval_steps = args.eval_steps
    tr_args.save_strategy = "steps"
    tr_args.save_steps = args.save_steps

    # Compatibility patch: some TRL vLLM servers require 'client_device_uuid' in
    # the init_communicator POST body, and using '0.0.0.0' for host can break NCCL
    # rendezvous. Intercept requests calls to add 'client_device_uuid' and replace
    # host with '127.0.0.1' when needed.
    try:
        # Patch requests (both top-level and Session)
        from requests import api as _api, sessions as _sessions  # type: ignore
        _orig_api_request = _api.request
        _orig_sess_request = _sessions.Session.request

        def _ensure_payload(kwargs: dict) -> None:
            j = kwargs.get("json")
            if isinstance(j, dict):
                j = dict(j)
                if "client_device_uuid" not in j:
                    j["client_device_uuid"] = str(uuid.uuid4())
                if j.get("host") in (None, "", "0.0.0.0"):
                    j["host"] = "127.0.0.1"
                kwargs["json"] = j

        def _inject_payload(method, url, *p_args, **p_kwargs):  # type: ignore[override]
            try:
                if isinstance(url, str) and url.rstrip("/").endswith("/init_communicator"):
                    _ensure_payload(p_kwargs)
            except Exception:
                pass
            return _orig_api_request(method, url, *p_args, **p_kwargs)

        def _sess_inject_payload(self, method, url, *p_args, **p_kwargs):  # type: ignore[override]
            try:
                if isinstance(url, str) and url.rstrip("/").endswith("/init_communicator"):
                    _ensure_payload(p_kwargs)
            except Exception:
                pass
            return _orig_sess_request(self, method, url, *p_args, **p_kwargs)

        _api.request = _inject_payload  # type: ignore[assignment]
        _sessions.Session.request = _sess_inject_payload  # type: ignore[assignment]
    except Exception:
        pass

    try:
        # Patch httpx client (post and generic request)
        import httpx  # type: ignore
        _orig_httpx_post = httpx.Client.post
        _orig_httpx_request = httpx.Client.request

        def _httpx_post(self, url, *p_args, **p_kwargs):  # type: ignore[override]
            try:
                if isinstance(url, str) and url.rstrip("/").endswith("/init_communicator"):
                    _ensure_payload(p_kwargs)
            except Exception:
                pass
            return _orig_httpx_post(self, url, *p_args, **p_kwargs)

        def _httpx_request(self, method, url, *p_args, **p_kwargs):  # type: ignore[override]
            try:
                if isinstance(url, str) and url.rstrip("/").endswith("/init_communicator"):
                    _ensure_payload(p_kwargs)
            except Exception:
                pass
            return _orig_httpx_request(self, method, url, *p_args, **p_kwargs)

        httpx.Client.post = _httpx_post  # type: ignore[assignment]
        httpx.Client.request = _httpx_request  # type: ignore[assignment]
    except Exception:
        pass

    # Guard against vLLM API variations: some versions may not expose
    # 'num_background_tasks' in the expected endpoint. Fall back to 0.
    try:
        import verifiers.inference.vllm_client as _vclient  # type: ignore
        _orig_get_num_bg = getattr(_vclient.VLLMClient, "get_num_background_tasks", None)

        if callable(_orig_get_num_bg):
            def _safe_get_num_background_tasks(self):  # type: ignore
                try:
                    return _orig_get_num_bg(self)
                except Exception:
                    return 0

            _vclient.VLLMClient.get_num_background_tasks = _safe_get_num_background_tasks  # type: ignore[attr-defined]
    except Exception:
        pass

    # Create separate OpenAI client for generation (port 8001)
    # The TRL communicator stays on port 8000 for /get_world_size and /init_communicator
    # This prevents 404 errors when env sends generation requests
    generation_client = AsyncOpenAI(
        api_key=os.getenv('OPENAI_API_KEY', 'EMPTY'),
        base_url='http://localhost:8001/v1'  # OpenAI server for generation
    )
    
    # Patch environment to use the generation client instead of communicator client
    import verifiers.envs.environment as envmod
    _orig_get_model_response = envmod.Environment.get_model_response
    
    async def _route_to_generation_server(self, *args, **kwargs):
        # Replace client with our generation client bound to port 8001
        args_list = list(args)
        if args_list:
            args_list[0] = generation_client
        kwargs['client'] = generation_client
        return await _orig_get_model_response(self, *args_list, **kwargs)
    
    envmod.Environment.get_model_response = _route_to_generation_server
    
    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=env,
        args=tr_args,
    )
    trainer.train()


if __name__ == "__main__":
    main()
