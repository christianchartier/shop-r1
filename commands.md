# Commands Ledger (from now on)

## Terminal C (remote, tmux) — verify RL endpoints
curl -L -s -o /dev/null -w "WS %{http_code}\n" http://localhost:8000/get_world_size/
curl -L -s -o /dev/null -w " IC %{http_code}\n" http://localhost:8000/init_communicator/

## Terminal C2 (remote) — RL (recommended)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m pip install -U 'verifiers @ git+https://github.com/willccbb/verifiers@main'
python scripts/rl_train_grpo.py --model checkpoints/sft --dataset data/sft.jsonl --output_dir checkpoints/rl_shop_r1 --alpha 0.13 --beta 0.001 --dars_factor 500 --max_steps 300 --learning_rate 1e-7 --temperature 0.6 --per_device_batch_size 1 --num_generations 2 --grad_accum 2 --max_seq_len 8192

## Terminal C2 (remote) — RL (fallback if 422 persists; inject client_device_uuid)
python - << 'PY'
import uuid, requests, runpy, sys
_orig_post = requests.post
def post_with_uuid(url, *args, **kwargs):
    try:
        if url.rstrip('/').endswith('/init_communicator'):
            j = kwargs.get('json')
            if isinstance(j, dict) and 'client_device_uuid' not in j:
                j = dict(j); j['client_device_uuid'] = str(uuid.uuid4())
                kwargs['json'] = j
    except Exception:
        pass
    return _orig_post(url, *args, **kwargs)
requests.post = post_with_uuid
sys.argv = [
  'scripts/rl_train_grpo.py',
  '--model','checkpoints/sft',
  '--dataset','data/sft.jsonl',
  '--output_dir','checkpoints/rl_shop_r1',
  '--alpha','0.13','--beta','0.001',
  '--dars_factor','500','--max_steps','300',
  '--learning_rate','1e-7','--temperature','0.6',
  '--per_device_batch_size','1','--num_generations','2',
  '--grad_accum','2','--max_seq_len','8192',
]
runpy.run_path('scripts/rl_train_grpo.py', run_name='__main__')
PY

## Terminal C (remote, tmux) — switch to eval server after RL finishes
pkill -f "trl.scripts.vllm_serve.*Qwen/Qwen2.5-3B-Instruct" || true
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-3B-Instruct --host 0.0.0.0 --port 8000 --dtype auto --max-model-len 32768 --gpu-memory-utilization 0.90

## Terminal B (local) — port forward for eval
ssh -p 1234 -N -L 8000:localhost:8000 root@69.19.136.229 -v

## Terminal A (local) — strict eval
export OPENAI_API_KEY=EMPTY && export OPENAI_BASE_URL=http://localhost:8000/v1
vf-eval shop-r1 -m local-qwen -a '{"strict":true,"normalize_variants":false,"sim_threshold":0.75,"debug_rewards":true,"debug_logprobs":true,"w_self_certainty":0.13,"system_prompt":"Output only a single JSON object with exactly two top-level keys: rationale (string) and action (object with keys type, name, text). Allowed types: click, type_and_submit, terminate. Type rules: terminate -> name=\\"\\" and text=\\"\\"; click -> name!=\\"\\" and text=\\"\\"; type_and_submit -> name!=\\"\\" and text!=\\"\\". No markdown, no extra keys, no commentary."}' -S '{"logprobs":true,"top_logprobs":5,"temperature":0,"response_format":{"type":"json_object"}}' -n 2 -r 1 -v

## Option A — Upgrade verifiers (preferred)
python -m pip install -U 'verifiers @ git+https://github.com/willccbb/verifiers@main'
python scripts/rl_train_grpo.py --model checkpoints/sft --dataset data/sft.jsonl --output_dir checkpoints/rl_shop_r1 --alpha 0.13 --beta 0.001 --dars_factor 500 --max_steps 300 --learning_rate 1e-7 --temperature 0.6 --per_device_batch_size 1 --num_generations 2 --grad_accum 2 --max_seq_len 8192

## Option B — Inject client_device_uuid (fallback)
python - << 'PY'
import uuid, requests, runpy, sys
_orig_post = requests.post
def post_with_uuid(url, *args, **kwargs):
    try:
        if url.rstrip('/').endswith('/init_communicator'):
            j = kwargs.get('json')
            if isinstance(j, dict) and 'client_device_uuid' not in j:
                j = dict(j); j['client_device_uuid'] = str(uuid.uuid4())
                kwargs['json'] = j
    except Exception:
        pass
    return _orig_post(url, *args, **kwargs)
requests.post = post_with_uuid
sys.argv = [
  'scripts/rl_train_grpo.py',
  '--model','checkpoints/sft',
  '--dataset','data/sft.jsonl',
  '--output_dir','checkpoints/rl_shop_r1',
  '--alpha','0.13','--beta','0.001',
  '--dars_factor','500','--max_steps','300',
  '--learning_rate','1e-7','--temperature','0.6',
  '--per_device_batch_size','1','--num_generations','2',
  '--grad_accum','2','--max_seq_len','8192',
]
runpy.run_path('scripts/rl_train_grpo.py', run_name='__main__')
PY

## Terminal C2 (remote) — RL (diagnostic injector with logs)
python - << 'PY'
import sys, uuid, runpy
# Patch requests.api.request and requests.Session.request
try:
    import requests
    from requests import api, sessions
    _orig_api_request = api.request
    _orig_sess_request = sessions.Session.request
    def api_request(method, url, *args, **kwargs):
        if isinstance(url, str) and url.rstrip('/').endswith('/init_communicator'):
            j = kwargs.get('json'); print(f"[patch] requests.api.request {method} {url} json={j}", file=sys.stderr)
            if isinstance(j, dict) and 'client_device_uuid' not in j:
                j = dict(j); j['client_device_uuid'] = str(uuid.uuid4()); kwargs['json'] = j
                print("[patch] added client_device_uuid (requests.api)", file=sys.stderr)
        return _orig_api_request(method, url, *args, **kwargs)
    def sess_request(self, method, url, *args, **kwargs):
        if isinstance(url, str) and url.rstrip('/').endswith('/init_communicator'):
            j = kwargs.get('json'); print(f"[patch] requests.Session.request {method} {url} json={j}", file=sys.stderr)
            if isinstance(j, dict) and 'client_device_uuid' not in j:
                j = dict(j); j['client_device_uuid'] = str(uuid.uuid4()); kwargs['json'] = j
                print("[patch] added client_device_uuid (requests.Session)", file=sys.stderr)
        return _orig_sess_request(self, method, url, *args, **kwargs)
    api.request = api_request
    sessions.Session.request = sess_request
    print("[patch] requests hooks installed", file=sys.stderr)
except Exception as e:
    print(f"[patch] requests not patched: {e}", file=sys.stderr)

# Patch httpx.Client.post
try:
    import httpx
    _orig_httpx_post = httpx.Client.post
    def httpx_post(self, url, *args, **kwargs):
        if isinstance(url, str) and url.rstrip('/').endswith('/init_communicator'):
            j = kwargs.get('json'); print(f"[patch] httpx.Client.post {url} json={j}", file=sys.stderr)
            if isinstance(j, dict) and 'client_device_uuid' not in j:
                j = dict(j); j['client_device_uuid'] = str(uuid.uuid4()); kwargs['json'] = j
                print("[patch] added client_device_uuid (httpx)", file=sys.stderr)
        return _orig_httpx_post(self, url, *args, **kwargs)
    httpx.Client.post = httpx_post
    print("[patch] httpx hook installed", file=sys.stderr)
except Exception as e:
    print(f"[patch] httpx not patched: {e}", file=sys.stderr)

sys.argv = [
  'scripts/rl_train_grpo.py',
  '--model','checkpoints/sft',
  '--dataset','data/sft.jsonl',
  '--output_dir','checkpoints/rl_shop_r1',
  '--alpha','0.13','--beta','0.001',
  '--dars_factor','500','--max_steps','300',
  '--learning_rate','1e-7','--temperature','0.6',
  '--per_device_batch_size','1','--num_generations','2',
  '--grad_accum','2','--max_seq_len','8192',
]
runpy.run_path('scripts/rl_train_grpo.py', run_name='__main__')
PY

## Terminal C (remote, tmux) — NCCL debug + safer settings
export NCCL_DEBUG=WARN
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
# Restart TRL server after exporting the vars if it was running before
pkill -f "trl.scripts.vllm_serve.*Qwen/Qwen2.5-3B-Instruct" || true
export TRANSFORMERS_NO_TORCHVISION=1
python -m trl.scripts.vllm_serve --model Qwen/Qwen2.5-3B-Instruct --host 0.0.0.0 --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.20

## Terminal C2 (remote) — RL (client NCCL debug env)
export NCCL_DEBUG=WARN
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
python scripts/rl_train_grpo.py --model checkpoints/sft --dataset data/sft.jsonl --output_dir checkpoints/rl_shop_r1 --alpha 0.13 --beta 0.001 --dars_factor 500 --max_steps 300 --learning_rate 1e-7 --temperature 0.6 --per_device_batch_size 1 --num_generations 2 --grad_accum 2 --max_seq_len 8192

## Terminal C2 (remote) — Re-run RL after communicator patch (httpx+requests injection)
python scripts/rl_train_grpo.py --model checkpoints/sft --dataset data/sft.jsonl --output_dir checkpoints/rl_shop_r1 --alpha 0.13 --beta 0.001 --dars_factor 500 --max_steps 300 --learning_rate 1e-7 --temperature 0.6 --per_device_batch_size 1 --num_generations 2 --grad_accum 2 --max_seq_len 8192

## Terminal C2 (remote) — Manual POST sanity check (replace port if different)
uuid=$(python - <<'PY'
import uuid; print(uuid.uuid4())
PY
)
curl -s -L -X POST http://localhost:8000/init_communicator/ -H 'Content-Type: application/json' \
  -d "{\"host\":\"0.0.0.0\",\"port\":51216,\"world_size\":2,\"client_device_uuid\":\"$uuid\"}" -w " IC %{http_code}\n"

## Terminal C2 (remote) — Confirm updated script is in use
python - << 'PY'
import inspect, scripts.rl_train_grpo as m
print(m.__file__)
src = open(m.__file__, 'r', encoding='utf-8').read()
print('HAS_INJECT_PATCH', ('client_device_uuid' in src) and ('httpx.Client.request' in src))
PY

## Terminal C2 (remote) — RL (robust injector wrapper: uuid + host=127.0.0.1)
python - << 'PY'
import sys, uuid, runpy
# Patch requests
try:
    import requests
    from requests import api, sessions
    _orig_api_request = api.request
    _orig_sess_request = sessions.Session.request
    def _ensure(kwargs):
        j = kwargs.get('json')
        if isinstance(j, dict):
            j = dict(j)
            j.setdefault('client_device_uuid', str(uuid.uuid4()))
            if j.get('host') in (None, '', '0.0.0.0'):
                j['host'] = '127.0.0.1'
            kwargs['json'] = j
    def api_request(method, url, *a, **k):
        if isinstance(url, str) and url.split('?',1)[0].rstrip('/').endswith('/init_communicator'):
            _ensure(k)
        return _orig_api_request(method, url, *a, **k)
    def sess_request(self, method, url, *a, **k):
        if isinstance(url, str) and url.split('?',1)[0].rstrip('/').endswith('/init_communicator'):
            _ensure(k)
        return _orig_sess_request(self, method, url, *a, **k)
    api.request = api_request
    sessions.Session.request = sess_request
except Exception:
    pass
# Patch httpx
try:
    import httpx
    _orig_httpx_post = httpx.Client.post
    _orig_httpx_req = httpx.Client.request
    def _ensure(k):
        j = k.get('json')
        if isinstance(j, dict):
            j = dict(j)
            j.setdefault('client_device_uuid', str(uuid.uuid4()))
            if j.get('host') in (None, '', '0.0.0.0'):
                j['host'] = '127.0.0.1'
            k['json'] = j
    def httpx_post(self, url, *a, **k):
        if isinstance(url, str) and url.split('?',1)[0].rstrip('/').endswith('/init_communicator'):
            _ensure(k)
        return _orig_httpx_post(self, url, *a, **k)
    def httpx_req(self, method, url, *a, **k):
        if isinstance(url, str) and url.split('?',1)[0].rstrip('/').endswith('/init_communicator'):
            _ensure(k)
        return _orig_httpx_req(self, method, url, *a, **k)
    httpx.Client.post = httpx_post
    httpx.Client.request = httpx_req
except Exception:
    pass
sys.argv = [
  'scripts/rl_train_grpo.py',
  '--model','checkpoints/sft',
  '--dataset','data/sft.jsonl',
  '--output_dir','checkpoints/rl_shop_r1',
  '--alpha','0.13','--beta','0.001',
  '--dars_factor','500','--max_steps','300',
  '--learning_rate','1e-7','--temperature','0.6',
  '--per_device_batch_size','1','--num_generations','2',
  '--grad_accum','2','--max_seq_len','8192',
]
runpy.run_path('scripts/rl_train_grpo.py', run_name='__main__')
PY

## Terminal C & C2 — Diagnose GPUs and env
nvidia-smi -L
echo "C2 CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

## If you have 2+ GPUs (preferred fix): pin server/client to different GPUs
# Terminal C (tmux) — restart TRL on GPU 0
pkill -f "trl.scripts.vllm_serve.*Qwen/Qwen2.5-3B-Instruct" || true
export TRANSFORMERS_NO_TORCHVISION=1
CUDA_VISIBLE_DEVICES=0 NCCL_DEBUG=WARN NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 \
  python -m trl.scripts.vllm_serve --model Qwen/Qwen2.5-3B-Instruct --host 0.0.0.0 --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.20

# Terminal C2 — run RL on GPU 1
CUDA_VISIBLE_DEVICES=1 NCCL_DEBUG=WARN NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 \
  python - << 'PY'
import sys, uuid, runpy, requests
from requests import api, sessions
_orig_api_request = api.request
_orig_sess_request = sessions.Session.request
def _ensure(kwargs):
    j = kwargs.get('json')
    if isinstance(j, dict):
        j = dict(j); j.setdefault('client_device_uuid', str(uuid.uuid4()))
        if j.get('host') in (None, '', '0.0.0.0'):
            j['host'] = '127.0.0.1'
        kwargs['json'] = j
def api_request(method, url, *a, **k):
    if isinstance(url, str) and url.split('?',1)[0].rstrip('/').endswith('/init_communicator'):
        _ensure(k)
    return _orig_api_request(method, url, *a, **k)
def sess_request(self, method, url, *a, **k):
    if isinstance(url, str) and url.split('?',1)[0].rstrip('/').endswith('/init_communicator'):
        _ensure(k)
    return _orig_sess_request(self, method, url, *a, **k)
api.request = api_request
sessions.Session.request = sess_request
sys.argv = [
  'scripts/rl_train_grpo.py','--model','checkpoints/sft','--dataset','data/sft.jsonl',
  '--output_dir','checkpoints/rl_shop_r1','--alpha','0.13','--beta','0.001',
  '--dars_factor','500','--max_steps','300','--learning_rate','1e-7','--temperature','0.6',
  '--per_device_batch_size','1','--num_generations','2','--grad_accum','2','--max_seq_len','8192',
]
runpy.run_path('scripts/rl_train_grpo.py', run_name='__main__')
PY

## If only 1 GPU — attempt CPU/gloo communicator (may be unsupported; capture logs)
# Terminal C (tmux): try forcing gloo backend and restart server
export VF_VLLM_COMM_BACKEND=gloo
export VLLM_COMM_BACKEND=gloo
export TORCH_DISTRIBUTED_DEBUG=DETAIL
pkill -f "trl.scripts.vllm_serve.*Qwen/Qwen2.5-3B-Instruct" || true
export TRANSFORMERS_NO_TORCHVISION=1
python -m trl.scripts.vllm_serve --model Qwen/Qwen2.5-3B-Instruct --host 0.0.0.0 --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.20

# Terminal C2: run RL again (watch for backend change in logs)
export VF_VLLM_COMM_BACKEND=gloo
export VLLM_COMM_BACKEND=gloo
export TORCH_DISTRIBUTED_DEBUG=DETAIL
python scripts/rl_train_grpo.py --model checkpoints/sft --dataset data/sft.jsonl --output_dir checkpoints/rl_shop_r1 --alpha 0.13 --beta 0.001 --dars_factor 500 --max_steps 300 --learning_rate 1e-7 --temperature 0.6 --per_device_batch_size 1 --num_generations 2 --grad_accum 2 --max_seq_len 8192

## Single-GPU sanity fallback (no weight sync): training loop only
python scripts/rl_train_grpo.py --model checkpoints/sft --dataset data/sft.jsonl --output_dir checkpoints/rl_shop_r1_nocomm --alpha 0.13 --beta 0.0 --dars_factor 500 --max_steps 100 --learning_rate 1e-7 --temperature 0.6 --per_device_batch_size 1 --num_generations 2 --grad_accum 2 --max_seq_len 8192

## Terminal C2 (remote) — Create reusable RL injector runner
cat > scripts/rl_inject.py << 'PY'
import sys, uuid, runpy

def _ensure_payload(k):
    j = k.get('json')
    if isinstance(j, dict):
        j = dict(j)
        j.setdefault('client_device_uuid', str(uuid.uuid4()))
        if j.get('host') in (None, '', '0.0.0.0'):
            j['host'] = '127.0.0.1'
        k['json'] = j

# Patch requests
try:
    import requests  # noqa: F401
    from requests import api, sessions
    _orig_api_request = api.request
    _orig_sess_request = sessions.Session.request
    def api_request(method, url, *a, **k):
        if isinstance(url, str) and url.split('?',1)[0].rstrip('/').endswith('/init_communicator'):
            _ensure_payload(k)
        return _orig_api_request(method, url, *a, **k)
    def sess_request(self, method, url, *a, **k):
        if isinstance(url, str) and url.split('?',1)[0].rstrip('/').endswith('/init_communicator'):
            _ensure_payload(k)
        return _orig_sess_request(self, method, url, *a, **k)
    api.request = api_request
    sessions.Session.request = sess_request
except Exception:
    pass

# Patch httpx
try:
    import httpx  # noqa: F401
    _orig_httpx_request = httpx.Client.request
    def httpx_request(self, method, url, *a, **k):
        if isinstance(url, str) and url.split('?',1)[0].rstrip('/').endswith('/init_communicator'):
            _ensure_payload(k)
        return _orig_httpx_request(self, method, url, *a, **k)
    httpx.Client.request = httpx_request
except Exception:
    pass

# Delegate to the real trainer with original CLI args
sys.argv = ['scripts/rl_train_grpo.py'] + sys.argv[1:]
runpy.run_path('scripts/rl_train_grpo.py', run_name='__main__')
PY

## Terminal C2 (remote) — Run RL via injector (uuid + host fix applied)
python scripts/rl_inject.py --model checkpoints/sft --dataset data/sft.jsonl --output_dir checkpoints/rl_shop_r1 --alpha 0.13 --beta 0.001 --dars_factor 500 --max_steps 300 --learning_rate 1e-7 --temperature 0.6 --per_device_batch_size 1 --num_generations 2 --grad_accum 2 --max_seq_len 8192

## Fresh Start (2× GPUs, after reboot) — stay on RL blocker

### Terminal C (remote, tmux) — start TRL vLLM server on GPU 0
tmux new -s vllm -d
tmux send-keys -t vllm "bash -lc 'export TRANSFORMERS_NO_TORCHVISION=1 NCCL_DEBUG=WARN NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1; \nCUDA_VISIBLE_DEVICES=0 python -m trl.scripts.vllm_serve --model Qwen/Qwen2.5-3B-Instruct --host 0.0.0.0 --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.20'" C-m

### Terminal C2 (remote) — setup env, pull latest, pin versions
ssh -p 1234 root@62.169.159.154
mkdir -p /ephemeral && cd /ephemeral && rm -rf shop-r1 && git clone https://github.com/christianchartier/shop-r1.git && cd shop-r1
python3.11 -m venv .venv311 && source .venv311/bin/activate
python -m pip install -U pip && python -m pip uninstall -y torchvision || true
python -m pip install "transformers>=4.55,<5" "trl==0.21.0" "vllm==0.10.1.1" accelerate>=0.30 peft>=0.11 datasets>=2.19 verifiers requests
python -m pip install -e . && vf-install shop-r1

# Optional: synthesize data if missing
[ -f data/sft.jsonl ] || python environments/shop_r1/synthesize.py -o data/sft.jsonl -n 1000 --seed 7

### Terminal C2 (remote) — verify TRL endpoints, then run RL on GPU 1
curl -L -s -o /dev/null -w "WS %{http_code}\n" http://localhost:8000/get_world_size/
curl -L -s -o /dev/null -w " IC %{http_code}\n" http://localhost:8000/init_communicator/

# Choose model: use SFT if present, else fall back to base
MODEL=$( [ -d checkpoints/sft ] && echo checkpoints/sft || echo Qwen/Qwen2.5-3B-Instruct )

# Run RL with communicator (server on GPU0, trainer on GPU1)
CUDA_VISIBLE_DEVICES=1 NCCL_DEBUG=WARN NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1 \
  python scripts/rl_train_grpo.py --model "$MODEL" --dataset data/sft.jsonl --output_dir checkpoints/rl_shop_r1 \
  --alpha 0.13 --beta 0.001 --dars_factor 500 --max_steps 300 --learning_rate 1e-7 --temperature 0.6 \
  --per_device_batch_size 1 --num_generations 2 --grad_accum 2 --max_seq_len 8192

# If 422 client_device_uuid occurs, run via injector wrapper instead
CUDA_VISIBLE_DEVICES=1 NCCL_DEBUG=WARN NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1 \
  python scripts/rl_inject.py --model "$MODEL" --dataset data/sft.jsonl --output_dir checkpoints/rl_shop_r1 \
  --alpha 0.13 --beta 0.001 --dars_factor 500 --max_steps 300 --learning_rate 1e-7 --temperature 0.6 \
  --per_device_batch_size 1 --num_generations 2 --grad_accum 2 --max_seq_len 8192

## After Reboot — Evaluation connection fix (A/B/C)

### Terminal C (remote, tmux) — start OpenAI server on 8001 (keeps TRL on 8000)
tmux new -s openai -d
tmux send-keys -t openai "bash -lc 'python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-3B-Instruct --host 0.0.0.0 --port 8001 --dtype auto --max-model-len 32768 --gpu-memory-utilization 0.90'" C-m

### Terminal B (local) — port forward 8001
ssh -p 1234 -N -L 8001:localhost:8001 root@62.169.159.154 -v

### Terminal A (local) — set base URL, health check, eval
export OPENAI_API_KEY=EMPTY && export OPENAI_BASE_URL=http://localhost:8001/v1
curl -s -o /dev/null -w "HTTP %{http_code}\n" "$OPENAI_BASE_URL/health"
vf-eval shop-r1 -m local-qwen -a '{"strict":false,"normalize_variants":true,"sim_threshold":0.75,"debug_rewards":true,"debug_logprobs":true,"w_self_certainty":0.13}' -S '{"logprobs":true,"top_logprobs":5,"temperature":0.2,"response_format":{"type":"json_object"}}' -n 2 -r 1 -v

## Fresh 2×GPU Bootstrap (new rig) — recreate venv + servers

### Terminal C2 (remote) — install deps, clone, create venv
ssh -p 1234 root@62.169.159.154
apt-get update && apt-get install -y python3.11 python3.11-venv git tmux
mkdir -p /ephemeral && cd /ephemeral && rm -rf shop-r1 && git clone https://github.com/christianchartier/shop-r1.git && cd shop-r1
python3.11 -m venv .venv311 && source .venv311/bin/activate && python -m pip install -U pip && python -m pip uninstall -y torchvision || true
python -m pip install "transformers>=4.55,<5" "trl==0.21.0" "vllm==0.10.1.1" accelerate>=0.30 peft>=0.11 datasets>=2.19 requests 'verifiers @ git+https://github.com/willccbb/verifiers@main'
python -m pip install -e . && vf-install shop-r1
[ -f data/sft.jsonl ] || python environments/shop_r1/synthesize.py -o data/sft.jsonl -n 1000 --seed 7

### Terminal C (remote, tmux) — start TRL server on GPU 0 (RL)
tmux new -s vllm -d
tmux send-keys -t vllm "bash -lc 'export TRANSFORMERS_NO_TORCHVISION=1 NCCL_DEBUG=WARN NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1; \
CUDA_VISIBLE_DEVICES=0 python -m trl.scripts.vllm_serve --model Qwen/Qwen2.5-3B-Instruct --host 0.0.0.0 --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.20'" C-m

### Terminal C2 (remote) — run RL on GPU 1 (communicator should succeed)
cd /ephemeral/shop-r1 && source .venv311/bin/activate
curl -L -s -o /dev/null -w "WS %\{http_code\}\n" http://localhost:8000/get_world_size/
curl -L -s -o /dev/null -w " IC %\{http_code\}\n" http://localhost:8000/init_communicator/
MODEL=$( [ -d checkpoints/sft ] && echo checkpoints/sft || echo Qwen/Qwen2.5-3B-Instruct )
CUDA_VISIBLE_DEVICES=1 NCCL_DEBUG=WARN NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1 \
  python scripts/rl_train_grpo.py --model "$MODEL" --dataset data/sft.jsonl --output_dir checkpoints/rl_shop_r1 \
  --alpha 0.13 --beta 0.001 --dars_factor 500 --max_steps 300 --learning_rate 1e-7 --temperature 0.6 \
  --per_device_batch_size 1 --num_generations 2 --grad_accum 2 --max_seq_len 8192

### Terminal C (remote, tmux) — OpenAI server for evaluation (optional; separate session/port 8001)
tmux new -s openai -d
tmux send-keys -t openai "bash -lc 'source /ephemeral/shop-r1/.venv311/bin/activate && \
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-3B-Instruct --host 0.0.0.0 --port 8001 --dtype auto --max-model-len 32768 --gpu-memory-utilization 0.90'" C-m

### Terminal B (local) — port forward eval (8001)
ssh -p 1234 -N -L 8001:localhost:8001 root@62.169.159.154 -v

### Terminal A (local) — health + eval
curl -s -o /dev/null -w "HTTP %{http_code}\n" http://localhost:8001/health
export OPENAI_API_KEY=EMPTY && export OPENAI_BASE_URL=http://localhost:8001/v1
vf-eval shop-r1 -m local-qwen -a '{"strict":false,"normalize_variants":true,"sim_threshold":0.75,"debug_rewards":true,"debug_logprobs":true,"w_self_certainty":0.13}' -S '{"logprobs":true,"top_logprobs":5,"temperature":0.2,"response_format":{"type":"json_object"}}' -n 2 -r 1 -v
