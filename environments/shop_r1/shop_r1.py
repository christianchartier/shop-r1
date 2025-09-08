"""
Shop-R1 environment for Verifiers and the Prime Environments Hub.

Production-oriented SingleTurnEnv approximating the reward design in:
"Shop‑R1: Rewarding LLMs to Simulate Human Behavior in Online Shopping via RL".

Features
- Strict JSON parser for rationale + action {type,name,value}.
- Hierarchical rewards (format, rationale, action type, attribute, value similarity).
- Self‑certainty reward using token logprobs when available.
- Difficulty-Aware Reward Scaling (DARS) that scales rewards by per-example difficulty.
- Dataset loading from JSONL or built-in examples. JSONL lines should be objects with
  keys: "prompt" (string or chat list) and "answer" (action dict), or "action".
"""

from __future__ import annotations

import json
import difflib
import os
import math
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from importlib import import_module

import verifiers as vf

# Optional Hugging Face datasets fallback
try:
    from datasets import Dataset as HFDataset  # type: ignore
except Exception:
    HFDataset = None  # type: ignore


# ---------------------------------------------------------------------------
# Built-in minimal examples (replace with your dataset)

EXAMPLES: List[Dict[str, Any]] = [
    {
        "prompt": [
            {
                "role": "user",
                "content": (
                    "Context: <html><body><h1>Search page</h1>"
                    "<input id='search_input'>Search</input></body></html>.\n"
                    "You are at the start of a shopping session. Provide a JSON"
                    " with your rationale and action."
                ),
            }
        ],
        "answer": {"type": "type_and_submit", "name": "search_input", "text": "laptop"},
    },
    {
        "prompt": [
            {
                "role": "user",
                "content": (
                    "Context: <html><body><h1>Search results for 'laptop'</h1>"
                    "<button id='add_to_cart'>Add to Cart</button></body></html>.\n"
                    "Previous actions: type_and_submit(search_input='laptop').\n"
                    "Provide a JSON with your rationale and action."
                ),
            }
        ],
        "answer": {"type": "click", "name": "add_to_cart"},
    },
    {
        "prompt": [
            {
                "role": "user",
                "content": (
                    "Context: <html><body><h1>Cart</h1><p>Your cart contains 1 item."\
                    "</p><button id='checkout'>Checkout</button></body></html>.\n"
                    "Previous actions: type_and_submit(search_input='laptop'), click(add_to_cart).\n"
                    "Provide a JSON with your rationale and action."
                ),
            }
        ],
        "answer": {"type": "terminate"},
    },
    {
        "prompt": [
            {
                "role": "user",
                "content": (
                    "Context: <html><body><h1>Search page</h1>"
                    "<input id='search_input'>Search</input></body></html>.\n"
                    "You are at the start of a shopping session. Provide a JSON"
                    " with your rationale and action."
                ),
            }
        ],
        "answer": {"type": "type_and_submit", "name": "search_input", "text": "headphones"},
    },
    {
        "prompt": [
            {
                "role": "user",
                "content": (
                    "Context: <html><body><h1>Search results for 'headphones'</h1>"
                    "<button id='view_details'>View Details</button>"
                    "<button id='add_to_cart'>Add to Cart</button></body></html>.\n"
                    "Previous actions: type_and_submit(search_input='headphones').\n"
                    "Provide a JSON with your rationale and action."
                ),
            }
        ],
        "answer": {"type": "click", "name": "view_details"},
    },
]


# ---------------------------------------------------------------------------
# Utilities

def rouge_l(pred: str, tgt: str) -> float:
    pred = (pred or "").strip().lower()
    tgt = (tgt or "").strip().lower()
    if not pred or not tgt:
        return 0.0
    matcher = difflib.SequenceMatcher(None, pred, tgt)
    match = sum(block.size for block in matcher.get_matching_blocks())
    return match / max(len(pred), len(tgt))


def _resolve(attr: str, candidates: list[tuple[str, str]], required: bool = True):
    """Best-effort resolver for verifiers symbols across versions.

    If not found and required is False, returns None instead of raising.
    """
    if hasattr(vf, attr):
        return getattr(vf, attr)
    for mod_name, name in candidates:
        try:
            mod = import_module(mod_name)
            if hasattr(mod, name):
                return getattr(mod, name)
        except Exception:
            continue
    if required:
        raise AttributeError(f"Unable to resolve '{attr}' from verifiers")
    return None


Dataset = _resolve(
    "Dataset",
    [
        ("verifiers.core", "Dataset"),
        ("verifiers.dataset", "Dataset"),
        ("verifiers.data", "Dataset"),
    ],
    required=False,
)

SingleTurnEnv = _resolve(
    "SingleTurnEnv",
    [
        ("verifiers.core", "SingleTurnEnv"),
        ("verifiers.environment", "SingleTurnEnv"),
    ],
    required=False,
)

# Optional MultiTurnEnv if the verifiers version exposes it
MultiTurnEnv = _resolve(
    "MultiTurnEnv",
    [
        ("verifiers.core", "MultiTurnEnv"),
        ("verifiers.environment", "MultiTurnEnv"),
    ],
    required=False,
)

Rubric = _resolve(
    "Rubric",
    [
        ("verifiers.core", "Rubric"),
        ("verifiers.rubric", "Rubric"),
    ],
    required=False,
)

ParserBase = _resolve(
    "Parser",
    [
        ("verifiers.core", "Parser"),
        ("verifiers.parser", "Parser"),
    ],
    required=False,
)

# ------------------ Fallback shims for local/offline testing -----------------
if ParserBase is None:  # minimal base for JSONActionParser
    class ParserBase:  # type: ignore
        pass

if Rubric is None:  # minimal container exposing funcs/weights
    class Rubric:  # type: ignore
        def __init__(self, parser=None, funcs=None, weights=None):
            self.parser = parser
            self.funcs = funcs or []
            self.weights = weights or []

if SingleTurnEnv is None:  # minimal env to satisfy tests
    class SingleTurnEnv:  # type: ignore
        def __init__(self, dataset, parser, rubric, system_prompt: str = "", **kwargs):
            self.dataset = dataset
            self.parser = parser
            self.rubric = rubric
            self.system_prompt = system_prompt
            self.config = kwargs


class JSONActionParser(ParserBase):
    """Parse JSON with rationale + action.

    Expects an object like:
      {"rationale": str, "action": {"type": str, "name": str, "value": str}}
    """

    def _extract_text(self, completion: Any) -> str:
        # Normalize completion into a text string
        if isinstance(completion, str):
            return completion
        if isinstance(completion, dict):
            c = completion.get("content")
            if isinstance(c, str):
                return c
        if isinstance(completion, list):
            # Prefer the last assistant/content string
            for item in reversed(completion):
                if isinstance(item, dict):
                    c = item.get("content")
                    if isinstance(c, str):
                        return c
                elif isinstance(item, str):
                    return item
        return ""

    def _find_json(self, completion: Any) -> Optional[str]:
        text = self._extract_text(completion)
        if not text:
            return None
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        cand = text[start : end + 1]
        try:
            json.loads(cand)
            return cand
        except Exception:
            return None

    def _is_single_json_only(self, completion: Any) -> bool:
        """True if the entire completion is exactly one JSON object.

        Enforces the paper's strict requirement: output a single JSON object
        with no extra prose, code fences, or trailing text.
        """
        text = self._extract_text(completion)
        if not isinstance(text, str) or not text.strip():
            return False
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return False
        candidate = text[start : end + 1]
        if text.strip() != candidate:
            return False
        try:
            obj = json.loads(candidate)
        except Exception:
            return False
        return isinstance(obj, dict)

    def _parse_json(self, completion: Any) -> Optional[Dict[str, Any]]:
        cand = self._find_json(completion)
        if cand is None:
            return None
        try:
            return json.loads(cand)
        except Exception:
            return None

    def _normalize_obj(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        # Normalize alternate schemas into the canonical one used by rewards.
        out = dict(obj)
        # Accept next_action as alias for action
        action = out.get("action")
        if not isinstance(action, dict) and isinstance(out.get("next_action"), dict):
            action = dict(out.get("next_action", {}))
        if not isinstance(action, dict):
            action = {}

        # Extract fields from possible aliases
        name = action.get("name") or action.get("target") or action.get("element") or ""
        if isinstance(name, str) and name.startswith("#"):
            name = name[1:]
        text = action.get("text") or action.get("value") or ""

        # Determine type from multiple hints
        type_hint = action.get("type") or action.get("action") or ""
        type_hint = str(type_hint).lower()
        if type_hint in {"type_and_submit", "type", "enter_text", "input"}:
            act_type = "type_and_submit"
        elif type_hint in {"click", "press", "tap"}:
            act_type = "click"
        elif type_hint in {"terminate", "close", "exit"}:
            act_type = "terminate"
        elif action.get("type") == "interact":
            # Some schemas use {type: interact, action: type|click}
            inner = str(action.get("action", "")).lower()
            if inner in {"type", "enter_text"}:
                act_type = "type_and_submit"
            elif inner in {"click", "press", "tap"}:
                act_type = "click"
            else:
                act_type = ""
        else:
            act_type = ""

        # Build canonical action
        canon: Dict[str, Any] = {"type": act_type}
        if act_type == "terminate":
            canon["name"] = ""
            canon["text"] = ""
        else:
            canon["name"] = str(name) if isinstance(name, str) else ""
            canon["text"] = str(text) if isinstance(text, str) else ""

        out["action"] = canon

        # Normalize rationale: if object, pick a sensible string field
        rat = out.get("rationale")
        if isinstance(rat, dict):
            for k in ("justification", "reason", "rationale"):
                if isinstance(rat.get(k), str):
                    out["rationale"] = rat.get(k)
                    break
        return out

    def parse_answer(self, completion: Any) -> Optional[Dict[str, Any]]:
        obj = self._parse_json(completion)
        if obj is None:
            return None
        obj = self._normalize_obj(obj)
        action = obj.get("action")
        return action if isinstance(action, dict) else None

    def parse_rationale(self, completion: Any) -> str:
        obj = self._parse_json(completion) or {}
        obj = self._normalize_obj(obj)
        r = obj.get("rationale")
        return r if isinstance(r, str) else ""


# ---------------------------------------------------------------------------
# Strict-mode helpers (paper-faithful schema enforcement)

ALLOWED_TYPES = {"click", "type_and_submit", "terminate"}


def _is_strict_top(obj: Dict[str, Any]) -> bool:
    return isinstance(obj, dict) and set(obj.keys()) == {"rationale", "action"}


def _is_strict_action_obj(act: Dict[str, Any]) -> bool:
    return isinstance(act, dict) and set(act.keys()) == {"type", "name", "text"}


def _is_strict_action_semantics(act: Dict[str, Any]) -> bool:
    t = act.get("type")
    name = act.get("name")
    text = act.get("text")
    if t not in ALLOWED_TYPES or not isinstance(name, str) or not isinstance(text, str):
        return False
    if t == "terminate":
        return name == "" and text == ""
    if t == "click":
        return (name != "") and (text == "")
    if t == "type_and_submit":
        return (name != "") and (text != "")
    return False

    def get_format_reward_func(self):
        def format_reward(parser: JSONActionParser, completion: str, answer: Dict[str, Any], **_) -> float:
            obj = parser._parse_json(completion)
            if obj is None:
                return 0.0
            if not isinstance(obj.get("rationale"), str):
                return 0.0
            act = obj.get("action")
            if not isinstance(act, dict):
                return 0.0
            if not isinstance(act.get("type"), str):
                return 0.0
            # Optional fields may be missing, but types must be strings if present
            for k in ("name", "text"):
                if k in act and not isinstance(act[k], str):
                    return 0.0
            return 1.0

        return format_reward


# ---------------------------------------------------------------------------
# Config + advanced reward helpers (DARS and self‑certainty)

@dataclass
class ShopR1Config:
    # Reward weights
    w_format: float = 0.5
    w_rationale: float = 0.13
    w_type: float = 0.3
    # Sub-action presence weights differ by action:
    #   click: +0.2 (if name != empty)
    #   type_and_submit: +0.1 (if name != empty) +0.1 (if text != empty)
    w_click_attr_presence: float = 0.2
    w_type_submit_name_presence: float = 0.1
    w_type_submit_text_presence: float = 0.1
    # Name similarity weight for type_and_submit (0.1 x ROUGE-L(name))
    w_type_submit_name_sim: float = 0.1
    # Optional self‑certainty term (0.13 in paper figure is the rationale reward)
    w_self_certainty: float = 0.13

    # DARS parameters
    enable_dars: bool = True
    dars_min_scale: float = 0.85
    dars_max_scale: float = 1.15
    # Additional DARS scaling factor for long-text sim (paper mentions 1000 during RL)
    dars_factor: float = 1000.0
    dars_weight_type: float = 0.4
    dars_weight_context_len: float = 0.3
    dars_weight_value_len: float = 0.3

    # Self‑certainty parameters
    enable_self_certainty: bool = True
    sc_calib_center: float = -2.5
    sc_calib_scale: float = 1.0
    sc_clip_min: float = 0.0
    sc_clip_max: float = 1.0
    # Similarity threshold (paper suggests 0.75)
    sim_threshold: float = 0.75

    # Dataset options
    dataset_path: Optional[str] = None
    max_examples: Optional[int] = None
    # Strict schema enforcement
    strict: bool = False
    # When not strict, optionally normalize variant schemas to canonical
    normalize_variants: bool = True
    # Debug flags
    debug_logprobs: bool = False
    debug_rewards: bool = False
    # Gate sub-rewards on correct type match (paper-faithful)
    gate_subrewards_on_type: bool = True


def _approx_tokens(text: str) -> int:
    return max(1, len((text or "").split()))


def _compute_dars_scale(cfg: ShopR1Config, prompt: List[Dict[str, str]], answer: Dict[str, Any]) -> float:
    if not cfg.enable_dars:
        return 1.0
    # Type difficulty heuristic
    type_w = {"terminate": 0.0, "click": 0.5, "type_and_submit": 1.0}
    t = answer.get("type", "")
    type_diff = type_w.get(t, 0.5)
    # Context difficulty by token count of user content
    ctx = " ".join(m.get("content", "") for m in prompt if m.get("role") == "user")
    ctx_len = _approx_tokens(ctx)
    ctx_diff = min(1.0, math.log(1 + ctx_len) / math.log(1 + 300))
    # Value difficulty by target length
    field = "text" if t == "type_and_submit" else "name"
    val_len = len((answer.get(field) or "").strip())
    val_diff = min(1.0, val_len / 64.0)
    total_w = cfg.dars_weight_type + cfg.dars_weight_context_len + cfg.dars_weight_value_len
    diff = (
        cfg.dars_weight_type * type_diff
        + cfg.dars_weight_context_len * ctx_diff
        + cfg.dars_weight_value_len * val_diff
    ) / (total_w if total_w > 0 else 1.0)
    return cfg.dars_min_scale + (cfg.dars_max_scale - cfg.dars_min_scale) * diff


def _extract_avg_logprob_from_kwargs(**kwargs) -> Optional[float]:
    # Common flat placements of token logprobs
    for key in ("completion_info", "meta", "extras"):
        info = kwargs.get(key)
        if isinstance(info, dict):
            for k in ("token_logprobs", "logprobs", "output_token_logprobs"):
                arr = info.get(k)
                if isinstance(arr, list) and arr:
                    # Case 1: list of floats
                    vals = [x for x in arr if isinstance(x, (int, float))]
                    if vals:
                        return sum(vals) / len(vals)
                    # Case 2: list of dicts with {'logprob': float}
                    if all(isinstance(x, dict) for x in arr):
                        vals = [x.get("logprob") for x in arr if isinstance(x.get("logprob"), (int, float))]
                        if vals:
                            return sum(vals) / len(vals)
                # Case 3: OpenAI-style object with content: [{token, logprob, top_logprobs}]
                if isinstance(arr, dict) and isinstance(arr.get("content"), list):
                    items = arr.get("content")
                    vals = [x.get("logprob") for x in items if isinstance(x, dict) and isinstance(x.get("logprob"), (int, float))]
                    if vals:
                        return sum(vals) / len(vals)
                if isinstance(arr, (int, float)):
                    return float(arr)
    for k in ("token_logprobs", "logprobs", "avg_logprob"):
        arr = kwargs.get(k)
        if isinstance(arr, list) and arr:
            vals = [x for x in arr if isinstance(x, (int, float))]
            if vals:
                return sum(vals) / len(vals)
        if isinstance(arr, (int, float)):
            return float(arr)
    # Inspect nested structures used by verifiers/openai clients
    def _as_dict(x):
        # Best-effort to convert model objects to dict-like access
        if isinstance(x, dict):
            return x
        # Try pydantic v2 BaseModel
        for fn in ("model_dump", "dict"):
            try:
                if hasattr(x, fn):
                    d = getattr(x, fn)()
                    if isinstance(d, dict):
                        return d
            except Exception:
                pass
        return None

    def _avg_lp_from_choice(choice) -> Optional[float]:
        # Accept dict or object with attributes
        lp = None
        if isinstance(choice, dict):
            lp = choice.get("logprobs")
        else:
            lp = getattr(choice, "logprobs", None)
        if lp is None:
            return None
        # OpenAI logprobs object has .content (list of tokens)
        content = None
        if isinstance(lp, dict):
            content = lp.get("content") or lp.get("tokens")
        else:
            content = getattr(lp, "content", None) or getattr(lp, "tokens", None)
        vals: list[float] = []
        if isinstance(content, list) and content:
            for tok in content:
                if isinstance(tok, dict):
                    v = tok.get("logprob")
                    if isinstance(v, (int, float)):
                        vals.append(float(v))
                else:
                    v = getattr(tok, "logprob", None)
                    if isinstance(v, (int, float)):
                        vals.append(float(v))
        # Some providers may put list of floats directly
        if not vals:
            if isinstance(lp, list):
                vals = [float(v) for v in lp if isinstance(v, (int, float))]
        if vals:
            return sum(vals) / max(1, len(vals))
        return None

    def _avg_lp_from_completion(comp) -> Optional[float]:
        if comp is None:
            return None
        # Dict view
        d = _as_dict(comp) or {}
        choices = None
        if isinstance(d, dict) and "choices" in d:
            choices = d.get("choices")
        if choices is None and hasattr(comp, "choices"):
            choices = getattr(comp, "choices")
        if isinstance(choices, list) and choices:
            # average across first choice tokens
            return _avg_lp_from_choice(choices[0])
        return None

    for container_key in ("state", "task"):
        container = kwargs.get(container_key)
        if isinstance(container, dict):
            # Try completion first
            lp = _avg_lp_from_completion(container.get("completion"))
            if lp is not None:
                return lp
            # Try responses list
            resps = container.get("responses")
            if isinstance(resps, list):
                for r in resps:
                    lp = _avg_lp_from_completion(r)
                    if lp is not None:
                        return lp
    return None


def _extract_topk_logprobs_from_completion(comp) -> Optional[List[List[float]]]:
    """Extract per-token top-k logprobs (as floats) from an OpenAI-like completion.

    Returns a list of tokens; each token is a list of top-k logprobs.
    """
    def _as_dict(x):
        if isinstance(x, dict):
            return x
        for fn in ("model_dump", "dict"):
            try:
                if hasattr(x, fn):
                    d = getattr(x, fn)()
                    if isinstance(d, dict):
                        return d
            except Exception:
                pass
        return None

    d = _as_dict(comp) or {}
    choices = None
    if isinstance(d, dict) and "choices" in d:
        choices = d.get("choices")
    if choices is None and hasattr(comp, "choices"):
        choices = getattr(comp, "choices")
    if not (isinstance(choices, list) and choices):
        return None
    choice0 = choices[0]
    logprobs = None
    if isinstance(choice0, dict):
        logprobs = choice0.get("logprobs")
    else:
        logprobs = getattr(choice0, "logprobs", None)
    if logprobs is None:
        return None
    content = None
    if isinstance(logprobs, dict):
        content = logprobs.get("content") or logprobs.get("tokens")
    else:
        content = getattr(logprobs, "content", None) or getattr(logprobs, "tokens", None)
    if not (isinstance(content, list) and content):
        return None
    per_token: List[List[float]] = []
    for tok in content:
        tops = None
        if isinstance(tok, dict):
            tops = tok.get("top_logprobs")
        else:
            tops = getattr(tok, "top_logprobs", None)
        vals: List[float] = []
        if isinstance(tops, list) and tops:
            for t in tops:
                lp = None
                if isinstance(t, dict):
                    lp = t.get("logprob")
                else:
                    lp = getattr(t, "logprob", None)
                if isinstance(lp, (int, float)):
                    vals.append(float(lp))
        if vals:
            per_token.append(vals)
    return per_token or None


def _extract_topk_logprobs_from_kwargs(**kwargs) -> Optional[List[List[float]]]:
    # Inspect nested structures used by verifiers/openai clients
    for container_key in ("state", "task"):
        container = kwargs.get(container_key)
        if isinstance(container, dict):
            # Try completion first
            seq = _extract_topk_logprobs_from_completion(container.get("completion"))
            if seq:
                return seq
            # Try responses list
            resps = container.get("responses")
            if isinstance(resps, list):
                for r in resps:
                    seq = _extract_topk_logprobs_from_completion(r)
                    if seq:
                        return seq
    return None


def _norm_certainty_from_topk(per_token_topk: List[List[float]]) -> float:
    """Compute normalized certainty 1 - H/ln(K_eff), averaged across tokens.

    Approximates average KL(p || U) up to an additive constant by using
    1 - H(p)/ln(K_eff), where K_eff is the number of buckets (top-k plus
    an OTHER bucket for residual probability mass).
    """
    if not per_token_topk:
        return 0.0
    scores: List[float] = []
    for vals in per_token_topk:
        ps = [math.exp(lp) for lp in vals if isinstance(lp, (int, float))]
        s_top = sum(ps)
        s_top = min(max(s_top, 0.0), 1.0)
        residual = max(0.0, 1.0 - s_top)
        buckets = ps[:]
        if residual > 1e-8:
            buckets.append(residual)
        Z = sum(buckets) or 1.0
        buckets = [max(1e-12, p / Z) for p in buckets]
        H = -sum(p * math.log(p) for p in buckets)
        K_eff = float(len(buckets))
        if K_eff <= 1:
            scores.append(1.0)
            continue
        H_max = math.log(K_eff)
        c = 1.0 - (H / H_max if H_max > 0 else 1.0)
        c = min(1.0, max(0.0, c))
        scores.append(c)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _self_certainty_score(cfg: ShopR1Config, **kwargs) -> float:
    if not cfg.enable_self_certainty:
        return 0.0
    tk = _extract_topk_logprobs_from_kwargs(**kwargs)
    if tk:
        c = _norm_certainty_from_topk(tk)
        return float(min(cfg.sc_clip_max, max(cfg.sc_clip_min, c)))
    # Fallback: avg logprob → sigmoid (legacy behaviour)
    avg_logprob = _extract_avg_logprob_from_kwargs(**kwargs)
    if avg_logprob is None:
        return 0.0
    z = (avg_logprob - cfg.sc_calib_center) * cfg.sc_calib_scale
    s = 1.0 / (1.0 + math.exp(-z))
    return float(min(cfg.sc_clip_max, max(cfg.sc_clip_min, s)))


def _load_examples_from_jsonl(path: str, max_examples: Optional[int] = None) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            prompt = obj.get("prompt")
            if isinstance(prompt, str):
                prompt = [{"role": "user", "content": prompt}]
            if not isinstance(prompt, list):
                continue
            answer = obj.get("answer") or obj.get("action")
            if not isinstance(answer, dict):
                continue
            # normalize: prefer 'text' over 'value' for type_and_submit
            if answer.get("type") == "type_and_submit":
                if "value" in answer and "text" not in answer:
                    answer["text"] = answer.pop("value")
            # drop empty fields
            if "name" in answer and (answer["name"] or "").strip() == "":
                answer.pop("name", None)
            if "text" in answer and (answer["text"] or "").strip() == "":
                answer.pop("text", None)
            data.append({"prompt": prompt, "answer": answer})
            if max_examples is not None and len(data) >= max_examples:
                break
    return data


# ---------------------------------------------------------------------------
# Environment constructor

def load_environment(**kwargs) -> SingleTurnEnv:
    """Construct the Shop‑R1 SingleTurnEnv with DARS + self‑certainty options.

    Accepts optional kwargs to override weights and features:
      - dataset_path: path to JSONL dataset (or env SHOP_R1_DATASET)
      - max_examples: int (or env SHOP_R1_MAX_EXAMPLES)
      - enable_dars: bool, dars_min_scale/dars_max_scale, dars_weight_* floats
      - enable_self_certainty: bool, w_self_certainty > 0 to include the term
      - sc_calib_center/sc_calib_scale/sc_clip_min/sc_clip_max
      - w_format/w_rationale/w_type/w_attribute/w_value/w_self_certainty
      - strict: bool, enforce exact schema without normalization
      - sim_threshold: float, value similarity threshold (default 0.75)
    """
    # Pull strict/sim flags off kwargs so they don't leak into SingleTurnEnv
    strict_flag = bool(kwargs.pop("strict", False))
    sim_threshold = float(kwargs.pop("sim_threshold", 0.75))

    cfg = ShopR1Config(
        w_format=float(kwargs.get("w_format", 0.5)),
        w_rationale=float(kwargs.get("w_rationale", 0.13)),
        w_type=float(kwargs.get("w_type", 0.3)),
        w_click_attr_presence=float(kwargs.get("w_click_attr_presence", 0.2)),
        w_type_submit_name_presence=float(kwargs.get("w_type_submit_name_presence", 0.1)),
        w_type_submit_text_presence=float(kwargs.get("w_type_submit_text_presence", 0.1)),
        w_type_submit_name_sim=float(kwargs.get("w_type_submit_name_sim", 0.1)),
        w_self_certainty=float(kwargs.get("w_self_certainty", 0.13)),
        enable_dars=bool(kwargs.get("enable_dars", True)),
        dars_min_scale=float(kwargs.get("dars_min_scale", 0.85)),
        dars_max_scale=float(kwargs.get("dars_max_scale", 1.15)),
        dars_weight_type=float(kwargs.get("dars_weight_type", 0.4)),
        dars_weight_context_len=float(kwargs.get("dars_weight_context_len", 0.3)),
        dars_weight_value_len=float(kwargs.get("dars_weight_value_len", 0.3)),
        dars_factor=float(kwargs.get("dars_factor", 1000.0)),
        enable_self_certainty=bool(kwargs.get("enable_self_certainty", True)),
        sc_calib_center=float(kwargs.get("sc_calib_center", -2.5)),
        sc_calib_scale=float(kwargs.get("sc_calib_scale", 1.0)),
        sc_clip_min=float(kwargs.get("sc_clip_min", 0.0)),
        sc_clip_max=float(kwargs.get("sc_clip_max", 1.0)),
        sim_threshold=sim_threshold,
        strict=strict_flag,
        dataset_path=kwargs.get("dataset_path") or os.environ.get("SHOP_R1_DATASET"),
        max_examples=int(os.environ.get("SHOP_R1_MAX_EXAMPLES", kwargs.get("max_examples") or 0)) or None,
        normalize_variants=bool(kwargs.get("normalize_variants", True)),
        debug_logprobs=bool(kwargs.get("debug_logprobs", False)),
        debug_rewards=bool(kwargs.get("debug_rewards", False)),
        gate_subrewards_on_type=bool(kwargs.get("gate_subrewards_on_type", True)),
    )

    # Dataset: load and normalize to include string 'answer' and dict 'info'
    if cfg.dataset_path:
        try:
            raw_examples = _load_examples_from_jsonl(cfg.dataset_path, cfg.max_examples)
        except Exception:
            raw_examples = EXAMPLES
    else:
        raw_examples = EXAMPLES
    # normalize rows
    normalized_rows: List[Dict[str, Any]] = []
    for ex in raw_examples:
        prompt = ex.get("prompt")
        gt = ex.get("answer") or {}
        if isinstance(gt, dict) and gt.get("type") == "type_and_submit" and "text" not in gt and "value" in gt:
            gt["text"] = gt.pop("value")
        normalized_rows.append({
            "prompt": prompt,
            # keep 'answer' as a string to satisfy Pydantic in some builds
            "answer": "",
            # store structured ground truth in 'info'
            "info": gt if isinstance(gt, dict) else {},
        })
    def _build_dataset(rows: List[Dict[str, Any]]):
        if Dataset is not None:
            try:
                return Dataset.from_list(rows)
            except Exception:
                pass
        if HFDataset is not None:
            try:
                return HFDataset.from_list(rows)
            except Exception:
                pass
        # Fallback: simple list acts as a dataset for local tests
        return list(rows)

    dataset = _build_dataset(normalized_rows)

    parser = JSONActionParser()

    # Reward closures applying DARS and self‑certainty
    def _get_ans(answer: Any, extras: Dict[str, Any]) -> Dict[str, Any]:
        info = extras.get("info")
        if isinstance(info, dict):
            return info  # type: ignore
        if isinstance(info, str):
            try:
                obj = json.loads(info)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
        return answer if isinstance(answer, dict) else {}

    def _format_reward(completion: str, answer: Any, **extras) -> float:
        # Constant format reward (paper-faithful; no DARS scaling)
        obj = parser._parse_json(completion)  # type: ignore[attr-defined]
        if obj is None:
            if cfg.debug_rewards:
                t = parser._extract_text(completion)  # type: ignore[attr-defined]
                print("[shop-r1] debug_format: no JSON object found; text=", (t or "").strip()[:280])
            return 0.0
        if cfg.strict:
            # Enforce single-JSON-object constraint (no extra text)
            if not parser._is_single_json_only(completion):  # type: ignore[attr-defined]
                if cfg.debug_rewards:
                    t = parser._extract_text(completion)  # type: ignore[attr-defined]
                    print("[shop-r1] debug_format(strict): not single-JSON-only; text=", (t or "").strip()[:280])
                return 0.0
            act = obj.get("action", {})
            top_ok = _is_strict_top(obj)
            act_obj_ok = _is_strict_action_obj(act)
            sem_ok = _is_strict_action_semantics(act)
            if not (top_ok and act_obj_ok and sem_ok):
                if cfg.debug_rewards:
                    print(
                        "[shop-r1] debug_format(strict): checks:",
                        {"top_keys": set(obj.keys()),
                         "top_ok": top_ok,
                         "action_keys": set(act.keys()) if isinstance(act, dict) else None,
                         "action_obj_ok": act_obj_ok,
                         "sem_ok": sem_ok,
                         "type": act.get("type") if isinstance(act, dict) else None,
                         "name": act.get("name") if isinstance(act, dict) else None,
                         "text": act.get("text") if isinstance(act, dict) else None},
                    )
                return 0.0
        else:
            # Only normalize variants when explicitly allowed (development mode)
            if cfg.normalize_variants:
                obj = parser._normalize_obj(obj)
            if not isinstance(obj.get("rationale"), str):
                if cfg.debug_rewards:
                    print("[shop-r1] debug_format: missing/invalid rationale string; keys=", set(obj.keys()))
                return 0.0
            act = obj.get("action")
            if not isinstance(act, dict):
                if cfg.debug_rewards:
                    print("[shop-r1] debug_format: missing action object; obj keys=", set(obj.keys()))
                return 0.0
            if not isinstance(act.get("type"), str):
                if cfg.debug_rewards:
                    print("[shop-r1] debug_format: missing action.type string; action keys=", set(act.keys()))
                return 0.0
            for k in ("name", "text"):
                if k in act and not isinstance(act[k], str):
                    if cfg.debug_rewards:
                        print("[shop-r1] debug_format: action field not string:", k, type(act.get(k)))
                    return 0.0
        return 1.0

    def _rationale_reward(completion: str, answer: Any, **extras) -> float:
        # Self‑certainty via normalized entropy / avg‑KL proxy from top‑k logprobs
        sc = _self_certainty_score(cfg, **extras)
        if cfg.debug_logprobs:
            try:
                keys = sorted(list(extras.keys()))
            except Exception:
                keys = []
            print(f"[shop-r1] debug_logprobs: self_certainty={sc:.4f}, extras_keys={keys}")
            # Inspect common nesting points where providers stash metadata
            for k in ("task", "state"):
                v = extras.get(k)
                if isinstance(v, dict):
                    sub_keys = list(v.keys())
                    print(f"[shop-r1] debug_logprobs: extras['{k}'] keys=", sub_keys[:20])
                    # If completion info is nested, show its keys
                    ci = v.get("completion_info") or v.get("meta") or v.get("extras")
                    if isinstance(ci, dict):
                        print(f"[shop-r1] debug_logprobs: extras['{k}'].(completion_info|meta|extras) keys=", list(ci.keys())[:20])
                    # Inspect known slots used by verifiers
                    comp = v.get("completion")
                    if isinstance(comp, dict):
                        print("[shop-r1] debug_logprobs: extras['", k, "'].completion keys=", list(comp.keys())[:20])
                        # OpenAI-like shape may have choices[0].logprobs
                        ch = comp.get("choices")
                        if isinstance(ch, list) and ch and isinstance(ch[0], dict):
                            print("[shop-r1] debug_logprobs: completion.choices[0] keys=", list(ch[0].keys())[:20])
                            lp0 = ch[0].get("logprobs")
                            if lp0 is not None:
                                print("[shop-r1] debug_logprobs: completion.choices[0].logprobs present (type)", type(lp0))
                    resps = v.get("responses")
                    if isinstance(resps, list) and resps:
                        r0 = resps[0]
                        print("[shop-r1] debug_logprobs: extras['", k, "'].responses[0] type=", type(r0))
                        if isinstance(r0, dict):
                            print("[shop-r1] debug_logprobs: responses[0] keys=", list(r0.keys())[:20])
                            if "logprobs" in r0:
                                print("[shop-r1] debug_logprobs: responses[0].logprobs present (type)", type(r0.get("logprobs")))
        return sc

    def _action_type_reward(completion: str, answer: Any, **extras) -> float:
        ans = _get_ans(answer, extras)
        if cfg.strict:
            obj = parser._parse_json(completion) or {}
            act = obj.get("action", {})
            if not (_is_strict_action_obj(act) and _is_strict_action_semantics(act)):
                base = 0.0
            else:
                base = 1.0 if act.get("type") == ans.get("type") else 0.0
        else:
            pred = parser.parse_answer(completion) or {}
            base = 1.0 if pred.get("type") == ans.get("type") else 0.0
        prompt = extras.get("prompt") or []
        return base * _compute_dars_scale(cfg, prompt, ans)

    def _attribute_reward(completion: str, answer: Any, **extras) -> float:
        ans = _get_ans(answer, extras)
        true_type = ans.get("type")
        base = 0.0
        if cfg.strict:
            obj = parser._parse_json(completion) or {}
            act = obj.get("action", {})
            if not (_is_strict_action_obj(act) and _is_strict_action_semantics(act)):
                return 0.0
            pred_type = act.get("type")
            if cfg.gate_subrewards_on_type and (pred_type != true_type):
                if cfg.debug_rewards:
                    print(f"[shop-r1] debug_attr: gated by type mismatch pred={pred_type} true={true_type}")
                return 0.0
            if true_type == "click" and (act.get("name") or "").strip():
                base = cfg.w_click_attr_presence
            elif true_type == "type_and_submit":
                if (act.get("name") or "").strip():
                    base += cfg.w_type_submit_name_presence
                if (act.get("text") or "").strip():
                    base += cfg.w_type_submit_text_presence
            return base
        else:
            pred = parser.parse_answer(completion) or {}
            pred_type = pred.get("type")
            if cfg.gate_subrewards_on_type and (pred_type != true_type):
                if cfg.debug_rewards:
                    print(f"[shop-r1] debug_attr: gated by type mismatch pred={pred_type} true={true_type}")
                return 0.0
            if true_type == "click":
                base = cfg.w_click_attr_presence if (pred.get("name") or "").strip() else 0.0
            elif true_type == "type_and_submit":
                if (pred.get("name") or "").strip():
                    base += cfg.w_type_submit_name_presence
                if (pred.get("text") or "").strip():
                    base += cfg.w_type_submit_text_presence
            return base

    def _value_reward(completion: str, answer: Any, **extras) -> float:
        ans = _get_ans(answer, extras)
        t = ans.get("type")
        if t == "terminate":
            return 0.0
        def sim(a: str, b: str) -> float:
            s = rouge_l(a, b)
            return s if s >= cfg.sim_threshold else 0.0

        reward = 0.0
        if cfg.strict:
            obj = parser._parse_json(completion) or {}
            act = obj.get("action", {})
            if not (_is_strict_action_obj(act) and _is_strict_action_semantics(act)):
                return 0.0
            pred_type = act.get("type")
            if cfg.gate_subrewards_on_type and (pred_type != t):
                if cfg.debug_rewards:
                    print(f"[shop-r1] debug_value: gated by type mismatch pred={pred_type} true={t}")
                return 0.0
            if t == "click":
                r_name = sim((act.get("name") or "").strip(), (ans.get("name") or "").strip())
                prompt = extras.get("prompt") or []
                reward += cfg.dars_factor * _compute_dars_scale(cfg, prompt, ans) * r_name
            else:
                r_name = sim((act.get("name") or "").strip(), (ans.get("name") or "").strip())
                reward += cfg.w_type_submit_name_sim * r_name
                r_text = sim((act.get("text") or "").strip(), (ans.get("text") or "").strip())
                prompt = extras.get("prompt") or []
                reward += cfg.dars_factor * _compute_dars_scale(cfg, prompt, ans) * r_text
            return reward
        else:
            pred = parser.parse_answer(completion) or {}
            pred_type = pred.get("type")
            if cfg.gate_subrewards_on_type and (pred_type != t):
                if cfg.debug_rewards:
                    print(f"[shop-r1] debug_value: gated by type mismatch pred={pred_type} true={t}")
                return 0.0
            if t == "click":
                r_name = sim((pred.get("name") or "").strip(), (ans.get("name") or "").strip())
                prompt = extras.get("prompt") or []
                reward += cfg.dars_factor * _compute_dars_scale(cfg, prompt, ans) * r_name
            else:
                r_name = sim((pred.get("name") or "").strip(), (ans.get("name") or "").strip())
                reward += cfg.w_type_submit_name_sim * r_name
                r_text = sim((pred.get("text") or "").strip(), (ans.get("text") or "").strip())
                prompt = extras.get("prompt") or []
                reward += cfg.dars_factor * _compute_dars_scale(cfg, prompt, ans) * r_text
            return reward

    def _self_certainty_reward(parser: JSONActionParser, completion: str, answer: Dict[str, Any], **extras) -> float:
        avg_lp = _extract_avg_logprob_from_kwargs(**extras)
        return _self_certainty_score(cfg, avg_lp)

    # Note: self‑certainty is the rationale reward (weight w_rationale).
    funcs = [_format_reward, _rationale_reward, _action_type_reward, _attribute_reward, _value_reward]
    weights = [cfg.w_format, cfg.w_rationale, cfg.w_type, 1.0, 1.0]
    
    # Attach parser to rubric to align with newer verifiers API and avoid parser mismatch warnings
    try:
        rubric = Rubric(parser=parser, funcs=funcs, weights=weights)
    except TypeError:
        # Backward-compat for older Rubric signature without parser kwarg
        rubric = Rubric(funcs=funcs, weights=weights)
    # Ensure rubric exposes funcs/weights for local tests even if upstream class hides them
    try:
        if getattr(rubric, "funcs", None) is None:
            setattr(rubric, "funcs", funcs)
        if getattr(rubric, "weights", None) is None:
            setattr(rubric, "weights", weights)
    except Exception:
        pass

    system_prompt = kwargs.pop(
        "system_prompt",
        "Output only a single JSON object with exactly two top-level keys: "
        "rationale (string) and action (object with keys type, name, text). "
        "Allowed types: click, type_and_submit, terminate. Rules: terminate→name=\"\" & text=\"\"; "
        "click→name!=\"\" & text=\"\"; type_and_submit→name!=\"\" & text!=\"\". "
        "No markdown, no extra keys, no commentary.",
    )

    return SingleTurnEnv(
        dataset=dataset,
        parser=parser,
        rubric=rubric,
        system_prompt=system_prompt,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Multi‑turn environment constructor

def _load_episodes_from_jsonl(path: str, max_episodes: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("steps"), list):
                steps = []
                for s in obj["steps"]:
                    prompt = s.get("prompt")
                    ans = s.get("answer") or s.get("action") or {}
                    if isinstance(ans, dict) and ans.get("type") == "type_and_submit" and "text" not in ans and "value" in ans:
                        ans["text"] = ans.pop("value")
                    steps.append({"prompt": prompt, "answer": ans})
                rows.append({"steps": steps})
            else:
                # Fallback: treat single line as a one‑step episode
                prompt = obj.get("prompt") if isinstance(obj, dict) else None
                ans = obj.get("answer") if isinstance(obj, dict) else None
                if isinstance(ans, dict) and ans.get("type") == "type_and_submit" and "text" not in ans and "value" in ans:
                    ans["text"] = ans.pop("value")
                rows.append({"steps": [{"prompt": prompt, "answer": ans or {}}]})
            if max_episodes is not None and len(rows) >= max_episodes:
                break
    return rows


def load_multiturn_environment(**kwargs):  # -> MultiTurnEnv
    if MultiTurnEnv is None:
        raise RuntimeError("MultiTurnEnv is not available in this verifiers version.")

    # Reuse config & parser
    strict_flag = bool(kwargs.pop("strict", False))
    sim_threshold = float(kwargs.pop("sim_threshold", 0.75))
    cfg = ShopR1Config(
        w_format=float(kwargs.get("w_format", 0.5)),
        w_rationale=float(kwargs.get("w_rationale", 0.13)),
        w_type=float(kwargs.get("w_type", 0.3)),
        w_click_attr_presence=float(kwargs.get("w_click_attr_presence", 0.2)),
        w_type_submit_name_presence=float(kwargs.get("w_type_submit_name_presence", 0.1)),
        w_type_submit_text_presence=float(kwargs.get("w_type_submit_text_presence", 0.1)),
        w_type_submit_name_sim=float(kwargs.get("w_type_submit_name_sim", 0.1)),
        w_self_certainty=float(kwargs.get("w_self_certainty", 0.13)),
        enable_dars=bool(kwargs.get("enable_dars", True)),
        dars_min_scale=float(kwargs.get("dars_min_scale", 0.85)),
        dars_max_scale=float(kwargs.get("dars_max_scale", 1.15)),
        dars_weight_type=float(kwargs.get("dars_weight_type", 0.4)),
        dars_weight_context_len=float(kwargs.get("dars_weight_context_len", 0.3)),
        dars_weight_value_len=float(kwargs.get("dars_weight_value_len", 0.3)),
        dars_factor=float(kwargs.get("dars_factor", 1000.0)),
        enable_self_certainty=bool(kwargs.get("enable_self_certainty", True)),
        sc_calib_center=float(kwargs.get("sc_calib_center", -2.5)),
        sc_calib_scale=float(kwargs.get("sc_calib_scale", 1.0)),
        sc_clip_min=float(kwargs.get("sc_clip_min", 0.0)),
        sc_clip_max=float(kwargs.get("sc_clip_max", 1.0)),
        sim_threshold=sim_threshold,
        strict=strict_flag,
        dataset_path=kwargs.get("dataset_path") or os.environ.get("SHOP_R1_DATASET"),
        max_examples=int(os.environ.get("SHOP_R1_MAX_EPISODES", kwargs.get("max_episodes") or 0)) or None,
        normalize_variants=bool(kwargs.get("normalize_variants", True)),
        debug_logprobs=bool(kwargs.get("debug_logprobs", False)),
        debug_rewards=bool(kwargs.get("debug_rewards", False)),
        gate_subrewards_on_type=bool(kwargs.get("gate_subrewards_on_type", True)),
    )

    if cfg.dataset_path:
        try:
            episodes = _load_episodes_from_jsonl(cfg.dataset_path, cfg.max_examples)
        except Exception:
            # Fallback from built‑ins as single‑step episodes
            episodes = [{"steps": [{"prompt": ex.get("prompt"), "answer": ex.get("answer")}] } for ex in EXAMPLES]
    else:
        episodes = [{"steps": [{"prompt": ex.get("prompt"), "answer": ex.get("answer")}] } for ex in EXAMPLES]

    # Convert to Dataset/HF Dataset with normalized fields
    rows: List[Dict[str, Any]] = []
    for ep in episodes:
        steps = []
        for s in ep.get("steps", []):
            gt = s.get("answer") or {}
            if isinstance(gt, dict) and gt.get("type") == "type_and_submit" and "text" not in gt and "value" in gt:
                gt["text"] = gt.pop("value")
            steps.append({"prompt": s.get("prompt"), "answer": "", "info": gt})
        rows.append({"steps": steps})

    def _build_dataset(rows: List[Dict[str, Any]]):
        if Dataset is not None:
            try:
                return Dataset.from_list(rows)
            except Exception:
                pass
        if HFDataset is not None:
            try:
                return HFDataset.from_list(rows)
            except Exception:
                pass
        return list(rows)

    dataset = _build_dataset(rows)
    parser = JSONActionParser()

    # Same rubric per step
    def _get_ans(answer: Any, extras: Dict[str, Any]) -> Dict[str, Any]:
        info = extras.get("info") or {}
        if isinstance(info, dict) and info:
            return info
        return answer if isinstance(answer, dict) else {}

    # Reuse rewards from single‑turn constructor
    def _format_reward(completion: str, answer: Any, **extras) -> float:
        t = parser._extract_text(completion)
        if t is None:
            return 0.0
        obj = parser._parse_json(t)
        if obj is None:
            return 0.0
        if cfg.strict and not parser._is_single_json_only(t):
            return 0.0
        return 1.0

    # Delegate to single‑turn reward builders by calling load_environment internals would be ideal;
    # for now, rebuild rubric by invoking load_environment and grabbing its rubric configuration.
    # Construct a temporary single‑turn env to source funcs/weights consistently.
    tmp_env = load_environment(dataset_path=None, max_examples=3)
    try:
        tmp_rubric = getattr(tmp_env, "rubric", None)
    except Exception:
        tmp_rubric = None
    if tmp_rubric is None:
        raise RuntimeError("Unable to build rubric for multi‑turn environment.")
    funcs = getattr(tmp_rubric, "funcs", None) or []
    weights = getattr(tmp_rubric, "weights", None) or []
    try:
        rubric = Rubric(parser=parser, funcs=funcs, weights=weights)
    except TypeError:
        rubric = Rubric(funcs=funcs, weights=weights)
    try:
        if getattr(rubric, "funcs", None) is None:
            setattr(rubric, "funcs", funcs)
        if getattr(rubric, "weights", None) is None:
            setattr(rubric, "weights", weights)
    except Exception:
        pass

    system_prompt = kwargs.pop("system_prompt", None) or (
        "<IMPORTANT> Your task is to predict the next action and provide rationale based on the previous "
        "actions and context. Output a single JSON with keys rationale and action. Allowed types: click, "
        "type_and_submit, terminate. No extra text. </IMPORTANT>"
    )

    try:
        return MultiTurnEnv(
            dataset=dataset,
            parser=parser,
            rubric=rubric,
            system_prompt=system_prompt,
            **kwargs,
        )
    except TypeError as e:
        # Fallback shim if MultiTurnEnv is abstract in this verifiers build
        if "abstract class" in str(e):
            class _Shim:
                def __init__(self, dataset, parser, rubric, system_prompt):
                    self.dataset = dataset
                    self.parser = parser
                    self.rubric = rubric
                    self.system_prompt = system_prompt
            return _Shim(dataset, parser, rubric, system_prompt)
        raise
