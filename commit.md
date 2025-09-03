# Shop‑R1 — Progress Update (High‑Fidelity Implementation)

Date: 2025‑09‑02

Summary
- Implemented SFT and RL scaffolding with verifiers + TRL/vLLM, plus evaluation utilities and a minimal command ledger to operate the pipeline end‑to‑end.
- Brought the environment and strict parser in line with paper constraints (single JSON; {type,name,text}; strict gating; DARS; similarity rewards).
- Reached an RL integration blocker on single‑GPU instances due to TRL vLLM weight‑sync using NCCL across server (vLLM) and client (trainer) ranks, which NCCL rejects when both ranks map to the same device (“Duplicate GPU detected”).
- Added robust client–server compatibility patches (client_device_uuid injection; host fix 0.0.0.0→127.0.0.1) and NCCL debug/compat envs; validated server endpoints and communicator POST.
- To unblock progress, we’re switching to evaluation flow while we queue the multi‑GPU or backend changes required for RL weight‑sync.

What’s Implemented
- Environment
  - Strict JSON parsing and reward rubric aligned with paper (format, rationale, action type, attribute presence, ROUGE‑L similarity, DARS; type‑gating).
- Training
  - `scripts/sft_train.py`: SFT of rationale+action with correct masking and chat template; left‑truncation to max_seq_len.
  - `scripts/rl_train_grpo.py`: GRPO trainer wiring (model/tokenizer, env, hyperparams) with vLLM TRL communicator integration; includes defensive patches to avoid accelerate device moves; injector hooks for init_communicator payload.
- Inference/Serving
  - TRL vLLM server instructions (tmux) for RL weight updates; OpenAI vLLM server for evaluation.
- Evaluation
  - `scripts/eval_actions.py`: Computes exact‑match, action‑type accuracy, macro‑F1, and per‑type breakdown against the endpoint alias registry.
- Operations/Docs
  - `commands.md`: Minimal “from now on” ledger with pinned versions, TRL server checks, RL/Eval commands, and fallback injectors and NCCL debug paths.

Key Logs / Evidence
- TRL server routes: /get_world_size/ = 200; /init_communicator/ = 200 (after injector); 
- RL crash on NCCL init with PyNcclCommunicator: “Duplicate GPU detected: rank 1 and rank 0 both on CUDA device”.

Limitations / Blockers
- Single‑GPU NCCL rank formation across server and trainer is not supported by the current TRL vLLM extension; requires either multi‑GPU (pin server to GPU0, trainer to GPU1) or a communicator backend that supports same‑device multi‑rank (not available by default).
- Forcing gloo via env did not switch backend in extension; PyNcclCommunicator still invoked.

Workarounds In Place
- Client/server POST compatibility: inject `client_device_uuid` and rewrite `host` to `127.0.0.1` for init_communicator.
- NCCL debug + compat env exports present; reusable injector script `scripts/rl_inject.py` added in ledger.
- Evaluation path documented and ready to run (switch server to OpenAI endpoint; local port‑forward; strict eval recipes).

Next Steps
- Preferred: provision 2× GPUs and rerun RL (server on GPU0, trainer on GPU1).
- Alternative: prototype single‑GPU RL without in‑process weight sync (e.g., local HF generation or periodic checkpoint reload) with clear caveats on faithfulness to TRL design.
- Continue with evaluation and ablations to produce paper‑style metrics while RL infra is updated.

README Updates (applied)
- Mark SFT and RL scaffolding as implemented with notes about communicator requirements.
- Add “Known Issues” for single‑GPU NCCL limitation and provide evaluation‑only instructions.
- Reference `commands.md` as the operational ledger.

Switch to Evaluation (Unblocking)
- The repo is updated to show evaluation steps; use OpenAI vLLM server (not TRL), port‑forward from local, and run strict/non‑strict eval commands. See `commands.md` and README “Evaluation” section.

