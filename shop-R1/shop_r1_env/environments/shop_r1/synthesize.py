import argparse
import json
import os
import random
from typing import List, Dict, Any, Optional

import time
import requests


PRODUCT_QUERIES = [
    "laptop", "wireless headphones", "mechanical keyboard", "gaming mouse",
    "4k monitor", "usb c hub", "smartphone case", "bluetooth speaker",
]

CLICK_TARGETS = [
    ("add_to_cart", "Add to Cart"),
    ("view_details", "View Details"),
    ("checkout", "Checkout"),
    ("apply_filter", "Apply Filter"),
]


def make_context(page: str, details: str = "") -> str:
    if page == "search":
        return (
            "Context: <html><body><h1>Search page</h1>"
            "<input id='search_input'>Search</input>"
            "<button id='search_button'>Search</button>"
            "</body></html>.\n" + details
        )
    if page == "results":
        return (
            "Context: <html><body><h1>Search results</h1>"
            "<div class='grid'>"
            "<button id='view_details'>View Details</button>"
            "<button id='add_to_cart'>Add to Cart</button>"
            "</div></body></html>.\n" + details
        )
    if page == "cart":
        return (
            "Context: <html><body><h1>Cart</h1>"
            "<p>Your cart contains 1 item.</p>"
            "<button id='checkout'>Checkout</button>"
            "</body></html>.\n" + details
        )
    return "Context: <html><body><p>Unknown</p></body></html>.\n" + details


def synthesize(n: int, seed: int = 7) -> List[Dict[str, Any]]:
    rnd = random.Random(seed)
    data: List[Dict[str, Any]] = []
    for i in range(n):
        # Choose a template
        t = rnd.random()
        if t < 0.4:
            # type_and_submit on search
            q = rnd.choice(PRODUCT_QUERIES)
            prompt = [{
                "role": "user",
                "content": make_context("search", "You are at the start of a shopping session. Provide JSON."),
            }]
            answer = {"type": "type_and_submit", "name": "search_input", "text": q}
        elif t < 0.85:
            # click on results page
            target_id, _ = rnd.choice(CLICK_TARGETS)
            prev = f"Previous actions: type_and_submit(search_input='{rnd.choice(PRODUCT_QUERIES)}').\n"
            prompt = [{
                "role": "user",
                "content": make_context("results", prev + "Provide JSON with rationale and next action."),
            }]
            answer = {"type": "click", "name": target_id}
        else:
            # terminate on cart
            prev = "Previous actions: type_and_submit(...), click(add_to_cart).\n"
            prompt = [{
                "role": "user",
                "content": make_context("cart", prev + "Provide JSON with rationale and next action."),
            }]
            answer = {"type": "terminate"}
        data.append({"prompt": prompt, "answer": answer})
    return data


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _rationale_prompt(context: str, action: Dict[str, Any]) -> str:
    return (
        "You are given a simplified HTML context observed by a shopper and the action they took.\n"
        "Write a short first-person rationale explaining why you took that action.\n\n"
        f"Context:\n{context}\n\n"
        f"Action (JSON):\n{json.dumps(action, ensure_ascii=False)}\n\n"
        "Rationale:"
    )


def generate_rationales_openai(base_url: str, api_key: str, model: str, rows: List[Dict[str, Any]], temperature: float = 0.2, rate_limit_s: float = 0.0) -> List[Optional[str]]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    out: List[Optional[str]] = []
    url = base_url.rstrip("/") + "/v1/chat/completions"
    for i, row in enumerate(rows):
        # extract last user content
        msgs = row.get("prompt") or []
        context = ""
        for m in reversed(msgs):
            if m.get("role") == "user":
                context = m.get("content") or ""
                break
        action = row.get("answer") or {}
        body = {
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": "You produce concise first-person rationales."},
                {"role": "user", "content": _rationale_prompt(context, action)},
            ],
        }
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            out.append(text.strip())
        except Exception:
            out.append(None)
        if rate_limit_s:
            time.sleep(rate_limit_s)
    return out


def main():
    ap = argparse.ArgumentParser(description="Synthesize Shop-R1-style JSONL dataset")
    ap.add_argument("-o", "--output", required=True, help="Output JSONL path")
    ap.add_argument("-n", "--num", type=int, default=1000, help="Number of examples")
    ap.add_argument("--seed", type=int, default=7, help="Random seed")
    # Optional: generate rationales via OpenAI-compatible API (e.g., vLLM OpenAI server)
    ap.add_argument("--rationales", action="store_true", help="Generate rationales via API and include in JSONL")
    ap.add_argument("--rationales-base-url", type=str, default=os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE") or "http://localhost:8000/v1", help="OpenAI-compatible base URL (e.g., http://localhost:8000/v1)")
    ap.add_argument("--rationales-model", type=str, default=os.environ.get("SHOP_R1_RAT_MODEL") or "Qwen/Qwen2.5-3B-Instruct", help="Model name for rationale generation")
    ap.add_argument("--rationales-key-env", type=str, default="OPENAI_API_KEY", help="Env var holding API key (set to any string for local vLLM)")
    ap.add_argument("--rationales-temp", type=float, default=0.2, help="Sampling temperature for rationales")
    ap.add_argument("--rationales-rate-limit", type=float, default=0.0, help="Sleep seconds between requests")
    args = ap.parse_args()
    rows = synthesize(args.num, args.seed)
    if args.rationales:
        key = os.environ.get(args.rationales_key_env, "EMPTY")
        rats = generate_rationales_openai(
            base_url=args.rationales_base_url,
            api_key=key,
            model=args.rationales_model,
            rows=rows,
            temperature=args.rationales_temp,
            rate_limit_s=args.rationales_rate_limit,
        )
        for r, rat in zip(rows, rats):
            if rat:
                r["rationale"] = rat
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} examples to {args.output}")


if __name__ == "__main__":
    main()
