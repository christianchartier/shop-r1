import json
import os
import tempfile

import pytest


def _write_tiny_jsonl(path: str):
    rows = [
        {
            "prompt": [
                {"role": "user", "content": "<html><body><input id='search_input'></body></html>"}
            ],
            "answer": {"type": "type_and_submit", "name": "search_input", "text": "laptop"},
        },
        {
            "prompt": [
                {"role": "user", "content": "<html><body><button id='add_to_cart'>Add to Cart</button></body></html>"}
            ],
            "answer": {"type": "click", "name": "add_to_cart"},
        },
        {
            "prompt": [
                {"role": "user", "content": "<html><body>Cart</body></html>"}
            ],
            "answer": {"type": "terminate"},
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_single_turn_env_import_and_basic_rewards():
    from environments.shop_r1.shop_r1 import load_environment

    with tempfile.TemporaryDirectory() as td:
        ds = os.path.join(td, "tiny.jsonl")
        _write_tiny_jsonl(ds)
        env = load_environment(dataset_path=ds, max_examples=3, strict=True)

        # Simulate three completions: good JSON, wrong type, and malformed
        good = json.dumps({
            "rationale": "Search for a product.",
            "action": {"type": "type_and_submit", "name": "search_input", "text": "laptop"},
        })
        wrong_type = json.dumps({
            "rationale": "Click something.",
            "action": {"type": "click", "name": "search_input", "text": ""},
        })
        malformed = "I will click now"

        # Access rubric funcs via env.rubric (verifiers api); fall back if unavailable
        rubric = getattr(env, "rubric", None)
        assert rubric is not None, "rubric should exist"
        funcs = getattr(rubric, "funcs", None)
        assert isinstance(funcs, (list, tuple)) and funcs, "rubric funcs present"

        # Evaluate format reward (func 0) on good vs malformed
        fmt = funcs[0]
        assert isinstance(fmt(good, {}), float)
        assert fmt(malformed, {}) == 0.0

        # Evaluate action type reward (func 2) should be higher for matched type
        type_reward = funcs[2]
        # Extras supply ground truth via info
        extras = {"info": {"type": "type_and_submit", "name": "search_input", "text": "laptop"}, "prompt": []}
        r_good = type_reward(good, {}, **extras)
        r_wrong = type_reward(wrong_type, {}, **extras)
        assert r_good >= r_wrong


def test_multiturn_constructor_when_available():
    try:
        from environments.shop_r1.shop_r1 import load_multiturn_environment
    except Exception:
        pytest.skip("MultiTurnEnv not available in this verifiers version")

    with tempfile.TemporaryDirectory() as td:
        ds = os.path.join(td, "episodes.jsonl")
        # One 2‑step episode
        ep = {
            "steps": [
                {
                    "prompt": [{"role": "user", "content": "context t1"}],
                    "answer": {"type": "type_and_submit", "name": "search_input", "text": "laptop"},
                },
                {
                    "prompt": [{"role": "user", "content": "context t2"}],
                    "answer": {"type": "click", "name": "view_details"},
                },
            ]
        }
        with open(ds, "w", encoding="utf-8") as f:
            f.write(json.dumps(ep) + "\n")

        try:
            env = load_multiturn_environment(dataset_path=ds, max_episodes=1, strict=True)
        except RuntimeError as e:
            if "MultiTurnEnv is not available" in str(e):
                pytest.skip("MultiTurnEnv not available in this verifiers version")
            raise
        assert env is not None
