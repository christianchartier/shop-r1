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
                    " with your rationale and next action."
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
                    "Provide a JSON with your rationale and next action."
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
                    "Provide a JSON with your rationale and next action."
                ),
            }
        ],
        "answer": {"type": "terminate"},
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
)

Rubric = _resolve(
    "Rubric",
    [
        ("verifiers.core", "Rubric"),
        ("verifiers.rubric", "Rubric"),
    ],
)

ParserBase = _resolve(
    "Parser",
    [
        ("verifiers.core", "Parser"),
        ("verifiers.parser", "Parser"),
    ],
)


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
    return None


def _self_certainty_score(cfg: ShopR1Config, avg_logprob: Optional[float]) -> float:
    if (avg_logprob is None) or (not cfg.enable_self_certainty):
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
    """
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
        dars_factor=float(kwargs.get("dars_factor", 1.0)),
        enable_self_certainty=bool(kwargs.get("enable_self_certainty", True)),
        sc_calib_center=float(kwargs.get("sc_calib_center", -2.5)),
        sc_calib_scale=float(kwargs.get("sc_calib_scale", 1.0)),
        sc_clip_min=float(kwargs.get("sc_clip_min", 0.0)),
        sc_clip_max=float(kwargs.get("sc_clip_max", 1.0)),
        sim_threshold=float(kwargs.get("sim_threshold", 0.75)),
        dataset_path=kwargs.get("dataset_path") or os.environ.get("SHOP_R1_DATASET"),
        max_examples=int(os.environ.get("SHOP_R1_MAX_EXAMPLES", kwargs.get("max_examples") or 0)) or None,
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
        raise RuntimeError(
            "Verifiers SingleTurnEnv expects a dataset object. Please install 'datasets' (huggingface) or use a verifiers version exposing Dataset."
        )

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
        # Re-implement format check to avoid signature mismatches
        obj = parser._parse_json(completion)  # type: ignore[attr-defined]
        if obj is None:
            return 0.0
        obj = parser._normalize_obj(obj)
        if not isinstance(obj.get("rationale"), str):
            return 0.0
        act = obj.get("action")
        if not isinstance(act, dict):
            return 0.0
        if not isinstance(act.get("type"), str):
            return 0.0
        for k in ("name", "text"):
            if k in act and not isinstance(act[k], str):
                return 0.0
        prompt = extras.get("prompt") or []
        ans = _get_ans(answer, extras)
        return 1.0 * _compute_dars_scale(cfg, prompt, ans)

    def _rationale_reward(completion: str, answer: Any, **extras) -> float:
        # Self‑certainty via average KL(p || U) approximation from token logprobs
        # Prefer token-level distributions; fall back to avg logprob heuristic
        avg_lp = _extract_avg_logprob_from_kwargs(**extras)
        sc = _self_certainty_score(cfg, avg_lp)
        return sc

    def _action_type_reward(completion: str, answer: Any, **extras) -> float:
        pred = parser.parse_answer(completion) or {}
        ans = _get_ans(answer, extras)
        base = 1.0 if pred.get("type") == ans.get("type") else 0.0
        prompt = extras.get("prompt") or []
        return base * _compute_dars_scale(cfg, prompt, ans)

    def _attribute_reward(completion: str, answer: Any, **extras) -> float:
        pred = parser.parse_answer(completion) or {}
        ans = _get_ans(answer, extras)
        true_type = ans.get("type")
        base = 0.0
        if true_type == "click":
            base = cfg.w_click_attr_presence if (pred.get("name") or "").strip() else 0.0
        elif true_type == "type_and_submit":
            if (pred.get("name") or "").strip():
                base += cfg.w_type_submit_name_presence
            if (pred.get("text") or "").strip():
                base += cfg.w_type_submit_text_presence
        return base

    def _value_reward(completion: str, answer: Any, **extras) -> float:
        pred = parser.parse_answer(completion) or {}
        ans = _get_ans(answer, extras)
        t = ans.get("type")
        if t == "terminate":
            return 0.0
        # ROUGE-L similarities with thresholding
        def sim(a: str, b: str) -> float:
            s = rouge_l(a, b)
            return s if s >= cfg.sim_threshold else 0.0

        reward = 0.0
        if t == "click":
            # DARS × ROUGE-L(name)
            r_name = sim((pred.get("name") or "").strip(), (ans.get("name") or "").strip())
            prompt = extras.get("prompt") or []
            reward += cfg.dars_factor * _compute_dars_scale(cfg, prompt, ans) * r_name
        else:  # type_and_submit
            # 0.1 × ROUGE-L(name)
            r_name = sim((pred.get("name") or "").strip(), (ans.get("name") or "").strip())
            reward += cfg.w_type_submit_name_sim * r_name
            # DARS × ROUGE-L(text)
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
    
    rubric = Rubric(funcs=funcs, weights=weights)

    system_prompt = kwargs.pop(
        "system_prompt",
        "You are a shopping assistant tasked with simulating human behaviour. "
        "Given the page context and past actions, output a JSON object with two keys: "
        "'rationale' (your reasoning) and 'action' (an object with 'type', 'name' and 'text'). "
        "For 'terminate' actions leave 'name' and 'text' empty.",
    )

    return SingleTurnEnv(
        dataset=dataset,
        parser=parser,
        rubric=rubric,
        system_prompt=system_prompt,
        **kwargs,
    )
