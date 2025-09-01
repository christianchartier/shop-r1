# Shop‑R1 Environment

This package contains a **simplified** implementation of the Shop‑R1
reinforcement learning environment described in the paper
*Shop‑R1: Rewarding LLMs to Simulate Human Behavior in Online Shopping via
Reinforcement Learning*.  The environment is built using the
[Verifiers](https://github.com/willccbb/verifiers) library and is
structured for deployment on the Prime Environments Hub.

## Overview

Shop‑R1 decomposes human shopping behaviour into two sub‑tasks: **rationale
generation** and **action prediction**.  At each step the agent
observes a simplified HTML context, thinks about what to do next, and
outputs a JSON object with two keys:

```json
{
  "rationale": "…",
  "action": {
    "type": "click" | "type_and_submit" | "terminate",
    "name": "…",  // optional attribute (e.g., button name)
    "text": "…"   // optional long‑text (e.g., search query; only for type_and_submit)
  }
}
```

The environment follows the Shop‑R1 reward design: format correctness (+0.5),
self‑certainty of the rationale (+0.13), action type accuracy (+0.3), sub‑action
attribute presence (+0.2 for click; +0.1/+0.1 for type_and_submit name/text), and
similarity rewards with thresholded ROUGE‑L and difficulty‑aware scaling (DARS)【239087673610332†L170-L177】【239087673610332†L431-L437】.  See
`environments/shop_r1/shop_r1.py` for the concrete implementation.

## Files

```
shop_r1_env/
├── environments/
│   └── shop_r1/
│       └── shop_r1.py   # Environment implementation
├── pyproject.toml       # Package metadata for installation
└── README.md            # This document
```

### `shop_r1.py`

Defines a `JSONActionParser` to extract the rationale and action from
model completions and computes several verifiable rewards.  A
placeholder dataset (`EXAMPLES`) demonstrates the required format; in
practice you should replace this with your own extracted context/action
pairs from shopping logs【239087673610332†L318-L343】.  The `load_environment` function
returns a `vf.SingleTurnEnv` with an appropriate rubric for RL.

### `pyproject.toml`

Declares the package metadata and a dependency on `verifiers`.  After
installing this package into your Python environment you can call
`vf-install shop-r1` and `vf-eval shop-r1` to evaluate models.

## Usage

1. **Create and activate a virtual env:**

   ```bash
   cd shop-R1/shop_r1_env
   if command -v uv >/dev/null 2>&1; then uv venv .venv; else python3 -m venv .venv; fi
   source .venv/bin/activate
   ```

2. **Install dependencies:** ensure you have a recent Python (≥3.8).  We
   recommend using [uv](https://github.com/astral-sh/uv) as your
   package manager.  You can install the `verifiers` library with:

   ```bash
   # install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # add the verifiers package (installs all extras by default)
   uv pip install verifiers[all]
   ```

3. **Install the environment locally:** use `uv` in place of `pip` to
   install the package in editable mode, then register the environment
   by its name (`shop-r1`) with verifiers:

   ```bash
   uv pip install -e .
    # register the environment (name maps hyphen ↔ underscore)
   vf-install shop-r1
   ```

4. **Evaluate a model (built-in examples):**

   ```bash
   vf-eval shop-r1 -m gpt-4.1-mini -n 3 -r 2
   ```

## Strict Mode (Paper‑Faithful)

Strict mode enforces the exact output schema used in the paper and disables all normalization. Under strict mode, completions must be a single JSON object — no prose, no code fences — with:

- Top‑level keys exactly: {"rationale", "action"}
- Action keys exactly: {"type", "name", "text"}
- Allowed types: click, type_and_submit, terminate
- Semantics:
  - terminate → name == "" and text == ""
  - click → name != "" and text == ""
  - type_and_submit → name != "" and text != ""

Enable strict mode and set the value similarity threshold via env args:

```bash
vf-eval shop-r1 \
  -m gpt-4.1-mini \
  -a '{"strict": true, "normalize_variants": false, "sim_threshold": 0.75}'
```

In strict mode, any deviation (extra keys, missing keys, wrong types, wrong empty/non‑empty values, code fences) yields 0 reward.

Helpful configs for strict runs:
- `normalize_variants=false`: disable alias normalization; only canonical keys `{type,name,text}` pass.
- `gate_subrewards_on_type=true` (default): attribute/value rewards require correct `action.type`.
- `debug_rewards=true`: prints which strict check failed and present keys.
- `debug_logprobs=true`: prints detected avg logprob and where it was found.

### Getting Non‑Zero Rewards Under Strict

Models adhere best when you use structured outputs:

1) OpenAI structured outputs (json_schema)

OpenAI’s `response_format.json_schema` enforces shape and types. Recent API versions disallow `oneOf` within nested properties; keep semantics enforced by the environment’s strict checks and use the schema only for shape.

```bash
vf-eval shop-r1 \
  -m gpt-4o-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY \
  -n 2 -r 1 \
  -S '{"temperature":0,"max_tokens":160,"response_format":{"type":"json_schema","json_schema":{"name":"ShopR1Strict","strict":true,"schema":{"type":"object","additionalProperties":false,"required":["rationale","action"],"properties":{"rationale":{"type":"string"},"action":{"type":"object","additionalProperties":false,"required":["type","name","text"],"properties":{"type":{"type":"string","enum":["click","type_and_submit","terminate"]},"name":{"type":"string"},"text":{"type":"string"}}}}}}}' \
  -a '{"strict": true, "system_prompt": "Output only a single JSON object with exactly two top-level keys: rationale (string) and action (object with keys type, name, text). Allowed types: click, type_and_submit, terminate. Type rules: terminate → name=\"\" and text=\"\"; click → name!=\"\" and text=\"\"; type_and_submit → name!=\"\" and text!=\"\". No markdown, no extra keys, no commentary."}'
```

2) vLLM or other OpenAI‑compatible servers (json_object)

If `json_schema` isn’t supported, use `json_object` plus a strong system prompt and temperature 0.

```bash
vf-eval shop-r1 \
  -m Qwen/Qwen2.5-3B-Instruct -b http://localhost:8000/v1 -k OPENAI_API_KEY \
  -n 2 -r 1 \
  -S '{"temperature":0,"max_tokens":160,"response_format":{"type":"json_object"}}' \
  -a '{"strict": true, "normalize_variants": false, "system_prompt": "Output only a single JSON object with exactly two top-level keys: rationale (string) and action (object with keys type, name, text). Allowed types: click, type_and_submit, terminate. Type rules: terminate → name=\"\" and text=\"\"; click → name!=\"\" and text=\"\"; type_and_submit → name!=\"\" and text!=\"\". No markdown, no extra keys, no commentary."}'
```

Tips
- Keep temperature low (0–0.2) to reduce drift.
- Few‑shot anchors can further improve adherence; add minimal exemplars via the `few_shot` env arg.

Example few‑shot anchors (as system messages):

```json
[{"role":"system","content":"Example output: {\"rationale\":\"...\",\"action\":{\"type\":\"click\",\"name\":\"add_to_cart\",\"text\":\"\"}}"},
 {"role":"system","content":"Example output: {\"rationale\":\"...\",\"action\":{\"type\":\"type_and_submit\",\"name\":\"search_input\",\"text\":\"laptop\"}}"},
 {"role":"system","content":"Example output: {\"rationale\":\"...\",\"action\":{\"type\":\"terminate\",\"name\":\"\",\"text\":\"\"}}"}]
```

Then pass `-a '{"strict":true, "few_shot": <the JSON above>}'`.
 
Troubleshooting strict zeros
- If the model emits `{rationale, next_action}` or alias fields like `element_id`, `value`, `submit`, strict will fail. Prefer `response_format=json_schema` (if supported), `temperature: 0`, and strict few‑shot anchors.
- Use `debug_rewards=true` to see exactly which check failed and the keys present.
- If rationale reward is 0, verify token logprobs are returned and enable `debug_logprobs=true` to inspect extras.

### Short Name vs Module Path

This repo also exposes a convenience entrypoint so you can run either form:

- Short env id (requires installation):

  ```bash
  uv pip install -e .
  uv run vf-eval shop-r1-env -a '{"strict":true}'
  ```

- Direct module path (no install needed):

  ```bash
  PYTHONPATH="$PWD:$PWD/environments" uv run vf-eval environments.shop_r1 -a '{"strict":true}'
  ```

### Endpoint Registry (optional)

Avoid repeating `-b/-k` by creating `./configs/endpoints.py`:

```python
ENDPOINTS = {
  "openai": {"model": "gpt-4o-mini", "url": "https://api.openai.com/v1", "key": "OPENAI_API_KEY"},
  "local-qwen": {"model": "Qwen/Qwen2.5-3B-Instruct", "url": "http://localhost:8000/v1", "key": "OPENAI_API_KEY"}
}
```

Then run: `vf-eval shop-r1 -m openai -a '{"strict":true}'`.

5. **Generate a synthetic dataset (JSONL) and point the env to it:**

   ```bash
   shop-r1-synth -o data/synth.jsonl -n 1000 --seed 7
   export SHOP_R1_DATASET=$PWD/data/synth.jsonl
   vf-eval shop-r1 -m gpt-4.1-mini -n 10 -r 2
   ```

### vLLM + Qwen (with logprobs for self‑certainty)

To activate the self‑certainty reward, your inference backend must return token logprobs.

1) Launch vLLM’s OpenAI server with Qwen:

```bash
# On a GPU machine/pod
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --dtype auto --max-model-len 32768 --gpu-memory-utilization 0.90

export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_BASE=$OPENAI_BASE_URL  # some clients use this name
```

2) Verify logprobs work via curl:

```bash
curl -s $OPENAI_BASE_URL/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "logprobs": 5, "top_logprobs": 5,
    "messages": [{"role":"user","content":"Say hi"}]
  }' | jq .choices[0].logprobs
```

3) Run `vf-eval` against the OpenAI endpoint and request logprobs.

Depending on your `verifiers` version, pass request extras either via a CLI flag
or environment variable. Check `vf-eval -h` for the supported option names.

Examples (adjust to your CLI):

```bash
# Pattern A: pass extras JSON directly
vf-eval shop-r1 \
  -m openai \
  --model-id Qwen/Qwen2.5-3B-Instruct \
  --base-url $OPENAI_BASE_URL \
  --extra '{"logprobs":5,"top_logprobs":5,"temperature":0.6}' \
  -n 10 -r 2

# Pattern B: set an env var consumed by your provider wrapper
export VERIFIERS_OPENAI_EXTRA='{"logprobs":5,"top_logprobs":5,"temperature":0.6}'
vf-eval shop-r1 -m openai --model-id Qwen/Qwen2.5-3B-Instruct -n 10 -r 2
```

If the extras are plumbed, the self‑certainty term in the environment will be active.

Notes on logprob wiring
- The environment extracts token logprobs from OpenAI/vLLM ChatCompletion objects threaded by verifiers under `extras['state'].completion` or `extras['state'].responses[0]`.
- Enable `debug_logprobs=true` to print the detected avg_logprob and extras keys.

What is the "extras" flag?

- Purpose: pass provider‑specific request options that aren’t modeled as first‑class CLI flags. Here you use it to ask the provider to return token logprobs.
- Shape: JSON string. Common keys for OpenAI‑compatible endpoints: `logprobs`, `top_logprobs`, `temperature`, `max_tokens`, etc.
- Where to find it: run `vf-eval -h` and look for an option named `--extra`, `--provider-extra`, or a provider‑scoped variant like `--openai.extra`. Some versions also read an env var (e.g., `VERIFIERS_OPENAI_EXTRA`). Use whichever your `vf-eval` supports.
- How it flows: vf‑eval merges the JSON into the underlying API request payload. The provider returns token logprobs; verifiers attaches them to `completion_info` and the environment’s self‑certainty reward uses them.

### Generate SFT‑style rationales via API

You can synthesize rationales with any OpenAI‑compatible endpoint (including vLLM):

```bash
# Using the running vLLM server above
shop-r1-synth -o data/synth_with_rationales.jsonl -n 1000 \
  --rationales \
  --rationales-base-url $OPENAI_BASE_URL \
  --rationales-model Qwen/Qwen2.5-3B-Instruct \
  --rationales-key-env OPENAI_API_KEY \
  --rationales-temp 0.2
```

Each JSONL row will include a `rationale` field alongside `prompt` and `answer`.

6. **Publish to the hub:** follow the instructions in the Prime
   Environments Hub setup guide to run `prime env init` and
   `prime env push` from the project root.  Once pushed, others can
   install your environment with:

   ```bash
   uv tool install prime
   prime login
   prime env init
   prime env push
   prime env info <you>/shop-r1
   prime env install <you>/shop-r1
   ```

7. **Train with Prime RL:** add a section like the following to your
   orchestrator TOML:

   ```toml
   [environment]
   id = "<you>/shop-r1"

   # Optional: pass environment kwargs
   [environment.kwargs]
   # Paper-aligned defaults
   w_format = 0.5
   w_rationale = 0.13
   w_type = 0.3
   sim_threshold = 0.75
   enable_dars = true
   dars_factor = 1000
   ```

   Ensure the environment is installed on the training machine (`prime env install <you>/shop-r1`). If using a custom dataset, set `SHOP_R1_DATASET=/path/to/your.jsonl`. Then run `uv run rl` with your trainer/inference/orchestrator configs.

Additionally, configure your inference to return token logprobs so the self‑certainty
reward is active. For OpenAI-compatible providers, pass `{"logprobs":5,"top_logprobs":5}`
in the inference section (key names vary by launcher; consult your Prime RL config schema). The env
will read logprobs from `completion.choices[0].logprobs.content[*].logprob`.

## Extending the Environment

This implementation is intentionally minimal.  To more faithfully
replicate the Shop‑R1 setup you should:

* Populate `EXAMPLES` with real shopping contexts and actions.  The
  paper describes a proprietary corpus of 52K sessions where each
  recorded action has a rationale generated by Claude 3.5【239087673610332†L513-L531】.
* Replace the heuristic rationale reward with a self‑certainty signal
  computed from token‑level probabilities (KL divergence) as described
  in Eq. (3)【239087673610332†L170-L177】.
* Implement the difficulty‑aware reward scaling (DARS) and adjust
  the weights in the rubric accordingly【239087673610332†L431-L437】.
* Use `MultiTurnEnv` if you wish to simulate full sessions rather
  than individual steps; override `is_completed` and
  `env_response` accordingly.

Refer to the Shop‑R1 paper and the Verifiers documentation for
additional guidance on dataset preparation and reward design.

## Recent Findings

- Schema drift explains strict zeros: prompts saying “rationale and next action” push models to output `next_action`. We updated defaults to say “rationale and action”, but strict still requires `{type,name,text}`.
- Logprob paths: verifiers stores OpenAI/vLLM responses under `extras['state'].completion` or `extras['state'].responses[0]`; the env now reads both to compute self‑certainty.
- Type‑gated subrewards: attribute/value credit is now gated by predicted type to prevent reward leakage.
- DARS factor: paper‑aligned default is `1000`; it amplifies long‑text similarity when type matches.

## End-to-End on Prime via Web UI (excruciating detail)

This is the exact flow we used to stand up Qwen via vLLM and activate the self‑certainty reward, using the Prime Intellect website to configure SSH.

1) Prepare an SSH key locally (one time)
- Ensure a keypair exists and note your public key:
  - `head -1 ~/.ssh/id_rsa.pub`  (should start with `ssh-ed25519` or `ssh-rsa`)
- If you don’t have one, create it:
  - `ssh-keygen -t ed25519 -f ~/.ssh/prime_intellect -C "prime-intellect"`
  - `cp ~/.ssh/prime_intellect ~/.ssh/id_rsa && chmod 600 ~/.ssh/id_rsa`
  - `cp ~/.ssh/prime_intellect.pub ~/.ssh/id_rsa.pub && chmod 644 ~/.ssh/id_rsa.pub`

2) Deploy GPU instance in the web UI
- Pick a non‑spot provider (e.g., Hyperstack/Datacrunch non‑spot) to ensure key injection.
- GPU: A100 80GB x1. Image: PyTorch 2.5 CUDA 12.4. Disk: 100–200 GB.
- Paste the full one‑line public key into the SSH key field.
- Wait for the instance to show an SSH line such as `root@IP -p PORT`.

3) Terminal C (remote) — start vLLM on the GPU
- Clear any stale host key if you reused an IP:
  - `ssh-keygen -R '[IP]:PORT'`
- SSH to the instance (accept host key):
  - `ssh -o StrictHostKeyChecking=accept-new -p PORT USER@IP`
- Inside the instance, install and run vLLM (leave this running):
  - `pip install --upgrade pip && pip install vllm`
  - `python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-3B-Instruct --host 0.0.0.0 --port 8000 --dtype auto --max-model-len 32768 --gpu-memory-utilization 0.90`

4) Terminal B (local) — port forward (keep open)
- `ssh -p PORT -N -L 8000:localhost:8000 USER@IP`

5) Terminal A (local, venv) — verify + evaluate
- `export OPENAI_API_KEY=EMPTY`
- `export OPENAI_BASE_URL=http://localhost:8000/v1`
- Verify logprobs via curl (one line):
  - `curl -s "$OPENAI_BASE_URL/chat/completions" -H "Authorization: Bearer $OPENAI_API_KEY" -H "Content-Type: application/json" -d '{"model":"Qwen/Qwen2.5-3B-Instruct","logprobs":5,"top_logprobs":5,"messages":[{"role":"user","content":"Say hi"}]}' | jq '.choices[0].logprobs'`
- Evaluate with logprobs (self‑certainty active):
  - `PYTHONPATH="$PWD:$PWD/environments" vf-eval environments.shop_r1 -m "Qwen/Qwen2.5-3B-Instruct" -b "$OPENAI_BASE_URL" -k OPENAI_API_KEY -S '{"logprobs":true,"top_logprobs":5,"temperature":0.2,"response_format":{"type":"json_object"}}' -n 2 -r 1`

6) Strict JSON schema (to see non‑zero format reward)
- The paper enforces a strict response object. You can nudge models by lowering temperature (e.g., 0.2) and adding a strict system prompt via `-a`:
  - `-a '{"system_prompt":"Output only this JSON (no prose, no code fences): {\"rationale\":\"<string>\",\"action\":{\"type\":\"click|type_and_submit|terminate\",\"name\":\"<string-or-empty>\",\"text\":\"<string-or-empty>\"}}"}'`

7) Known_hosts + quoting tips
- For non‑22 ports, host keys are stored as `[IP]:PORT` in `~/.ssh/known_hosts`. Remove with: `ssh-keygen -R '[IP]:PORT'`.
- zsh requires quoting JSON and jq filters. Keep commands on one line; if you see `quote>` prompts, re‑paste without stray newlines.

8) Clean up
- Stop vLLM (Ctrl‑C in Terminal C). Stop port forward (Ctrl‑C in Terminal B). Terminate the instance from the web UI to stop billing.

## Paper fidelity vs. normalization

- This environment includes a small normalization layer to map common variants (e.g., `next_action.target → name`, `value → text`, CSS `#id → id`) into the canonical `{type,name,text}` for development ergonomics.
- For “paper‑strict” ablations, keep prompts strict so the model emits the exact schema; normalization does not alter correctly‑formed outputs.

## Easy Reboot (what worked reliably)

Once your SSH public key is saved in the web UI, you do NOT need to re‑provide it on each reboot. The following is the fastest flow we used:

- Terminal C (remote; after the instance shows an SSH line in the UI)
  - Click the SSH connection in the UI. Example shown:
    - `ssh root@<IP> -p <PORT>`
  - First connect (accept host key):
    - `ssh -o StrictHostKeyChecking=accept-new -p <PORT> root@<IP>`
  - Prepare vLLM and start the server (keep running):
    - `pip install --upgrade pip`
    - `pip install vllm`
    - (optional) `apt-get update && apt-get install -y tmux && tmux new -s vllm`
    - `python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-3B-Instruct \
        --host 0.0.0.0 --port 8000 \
        --dtype auto --max-model-len 32768 --gpu-memory-utilization 0.90`

- Terminal B (local Mac; keep open)
  - Forward port 8000 to your Mac (verbose helps if there’s an error):
    - `ssh -p <PORT> -N -L 8000:localhost:8000 root@<IP> -v`

- Terminal A (local Mac, venv active in this repo)
  - Point verifiers to the local forward and run eval with logprobs + self‑certainty:
    - `export OPENAI_API_KEY=EMPTY`
    - `export OPENAI_BASE_URL=http://localhost:8000/v1`
    - `PYTHONPATH="$PWD:$PWD/environments" vf-eval environments.shop_r1 -m "Qwen/Qwen2.5-3B-Instruct" -b "$OPENAI_BASE_URL" -k OPENAI_API_KEY -S '{"logprobs":true,"top_logprobs":5,"temperature":0.2,"response_format":{"type":"json_object"}}' -a '{"w_self_certainty":0.13}' -n 2 -r 1`

Notes
- If the UI shows a different SSH user (e.g., `ubuntu@`), use that exact user.
- If you reused an IP and see a host‑key warning, remove the old key: `ssh-keygen -R '[<IP>]:<PORT>'` and reconnect.


## TODOs (Paper‑complete implementation)

Use this checklist to complete a paper‑faithful implementation and reproductions.

- Strict schema + format reward
  - [x] Add a `strict` flag (default off; enable for paper runs) to reject any non‑canonical keys; normalization is bypassed when strict.
  - [x] Add a `normalize_variants` flag (default true; set false for paper runs) to map aliases only for development.
  - [x] Enforce “single JSON object, nothing else” (no prose/code fences) in strict mode.

- Self‑certainty reward (rationale)
  - [ ] Implement average KL(p || U) over rationale tokens (Eq. 3). If only top‑k is available, renormalize top‑k and add residual mass.
  - [ ] Ensure we isolate rationale tokens (not action) when computing s.
  - [x] Expose `w_self_certainty` and enable/disable via config.

- DARS (difficulty‑aware reward scaling)
  - [x] Defaults: `dars_factor = 1000` and weights for type/context/value difficulty; all exposed as config.
  - [x] Gate similarity with threshold 0.75; verify click vs type_and_submit reward paths.
  - [x] Gate subrewards by type match: `gate_subrewards_on_type=true`.

- Hierarchical rewards (Table 1)
  - [x] Type: +0.3 on exact match {click, type_and_submit, terminate}.
  - [x] Sub‑action presence: click +0.2 (name); type_and_submit +0.1 (name) +0.1 (text).
  - [x] Similarity: type_and_submit +0.1×ROUGE‑L(name) and DARS×ROUGE‑L(text); click DARS×ROUGE‑L(name).

- Multi‑turn environment (sessions)
  - [ ] Implement `MultiTurnEnv` with per‑episode state (page HTML, history), `env_response(...)`, and `is_completed(...)`.
  - [ ] Config for whole‑session vs latest‑step context (paper evaluates both).
  - [ ] Episode‑level aggregation if any delayed signals are required.
  - [ ] Session JSONL schema + converter and strict validator CLI.

- Dataset + loaders
  - [x] Define JSONL schema for step data `{prompt, answer{type,name?,text?}}` and loader.
  - [ ] Add session→steps converter; include simplified HTML and prior actions per step.
  - [ ] Validator CLI to check schema and required fields.

- Prompting + sampling
  - [x] Provide Appendix‑style strict prompt and minimal one‑shot examples per action.
  - [x] Default temp band guidance 0.6–0.8; set 0.6 for eval; allow overrides.
  - [x] Optional `response_format=json_object` where supported.

- Integration + configs
  - [x] Prime Hub: `prime env init/push` with versioned releases.
  - [x] Prime RL TOMLs with paper weights (α=0.005, β=0.001), DARS=1000, temp≈0.6.
  - [x] Ensure inference returns token logprobs/top_logprobs.

- Tests + CI
  - [ ] Unit tests: parser strict/normalize; format reward; hierarchical rewards; DARS; self‑certainty.
  - [ ] Mock provider test for logprobs path.
  - [ ] Type‑gated subrewards; strict schema failures; DARS scaling determinism.

- Logging + analysis
- [ ] Verbose mode to dump parsed JSON, normalized action, and per‑reward components.
- [ ] Optional W&B logging for reward breakdowns and self‑certainty.
 - [ ] `gate_all_on_format` option to zero all action rewards when format fails (for ablations).

## Power Down (stop billing) and Reboot Notes

Power down (when you’re done)
- Terminal C (remote): stop vLLM
  - If running in the foreground: press Ctrl‑C
  - If running in tmux: `tmux kill-session -t vllm`
- Terminal B (local): stop SSH port‑forward
  - Press Ctrl‑C
- Terminate the GPU instance (stops hourly charges)
  - Web UI: Instances → select your instance → Terminate
  - Or CLI: `prime pods terminate <pod-id>`

Reboot checklist (bring it back up later)
1) Deploy a fresh instance (or start a saved template) via the web UI. Ensure your SSH public key is set.
2) Terminal C (remote): SSH in (use the SSH line from the UI), then start vLLM
   - `pip install --upgrade pip && pip install vllm`
   - `python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-3B-Instruct --host 0.0.0.0 --port 8000 --dtype auto --max-model-len 32768 --gpu-memory-utilization 0.90`
3) Terminal B (local): port‑forward 8000
   - `ssh -p PORT -N -L 8000:localhost:8000 USER@IP`
4) Terminal A (local): point vf‑eval at the endpoint and run
   - `export OPENAI_API_KEY=EMPTY`
   - `export OPENAI_BASE_URL=http://localhost:8000/v1`
   - `PYTHONPATH="$PWD:$PWD/environments" vf-eval environments.shop_r1 -m "Qwen/Qwen2.5-3B-Instruct" -b "$OPENAI_BASE_URL" -k OPENAI_API_KEY -S '{"logprobs":true,"top_logprobs":5,"temperature":0.2,"response_format":{"type":"json_object"}}' -n 2 -r 1`

Host key and quoting tips
- If the IP is reused, clear the old known_hosts entry: `ssh-keygen -R '[IP]:PORT'`
- zsh requires quoting JSON and jq filters; keep commands on one line.
