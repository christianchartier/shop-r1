import argparse
import json
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from openai import OpenAI

import verifiers as vf
from environments.shop_r1.shop_r1 import JSONActionParser
from environments.shop_r1.shop_r1 import rouge_l  # reuse similarity


def load_dataset(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def canonical_action(ans: Dict[str, Any]) -> Dict[str, Any]:
    t = str(ans.get("type", "")).lower()
    out = {"type": t}
    if t == "terminate":
        out["name"] = ""
        out["text"] = ""
    else:
        out["name"] = str(ans.get("name") or ans.get("element") or ans.get("target") or "")
        out["text"] = str(ans.get("text") or ans.get("value") or "")
    return out


def metrics(rows: List[Tuple[Dict[str, Any], Dict[str, Any]]], sim_threshold: float = 0.75) -> Dict[str, Any]:
    y_true = []
    y_pred = []
    exact_hits = 0
    by_type = defaultdict(lambda: {"exact": 0, "count": 0, "type_hit": 0})

    for truth, pred in rows:
        t_type = truth.get("type")
        p_type = pred.get("type")
        y_true.append(t_type)
        y_pred.append(p_type)

        type_hit = int(t_type == p_type)
        by_type[t_type]["type_hit"] += type_hit
        by_type[t_type]["count"] += 1

        # exact-match
        ok = False
        if t_type == "terminate":
            ok = type_hit == 1
        elif t_type == "click":
            ok = type_hit == 1 and (truth.get("name") == (pred.get("name") or "").strip())
        elif t_type == "type_and_submit":
            name_ok = (truth.get("name") or "") == (pred.get("name") or "").strip()
            text_ok = rouge_l((pred.get("text") or "").strip(), (truth.get("text") or "").strip()) >= sim_threshold
            ok = type_hit == 1 and name_ok and text_ok
        if ok:
            exact_hits += 1
            by_type[t_type]["exact"] += 1

    # overall metrics
    exact_acc = exact_hits / max(1, len(rows))
    # type accuracy and macro F1
    labels = ["click", "type_and_submit", "terminate"]
    type_acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / max(1, len(y_true))
    f1s = []
    for c in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == c and b == c)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != c and b == c)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == c and b != c)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 0.0 if (prec + rec) == 0 else (2 * prec * rec / (prec + rec))
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s)

    per_type = {
        t: {
            "exact_action_acc": d["exact"] / max(1, d["count"]),
            "action_type_acc": d["type_hit"] / max(1, d["count"]),
            "count": d["count"],
        }
        for t, d in by_type.items()
    }
    return {
        "exact_action_acc": exact_acc,
        "action_type_acc": type_acc,
        "action_type_f1": macro_f1,
        "per_type": per_type,
        "n": len(rows),
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate exact-match and type metrics on Shop-R1")
    ap.add_argument("--dataset", required=True, help="JSONL with prompt+answer (ground truth)")
    ap.add_argument("--model_alias", default="local-qwen", help="Endpoint alias in configs/endpoints.py")
    ap.add_argument("--env_strict", action="store_true")
    ap.add_argument("--sim_threshold", type=float, default=0.75)
    ap.add_argument("--max_examples", type=int, default=-1)
    ap.add_argument("--out", default="eval_results.json")
    args = ap.parse_args()

    # Load env with dataset and parser
    vf_env = vf.load_environment(
        env_id="shop-r1",
        dataset_path=args.dataset,
        strict=args.env_strict,
        sim_threshold=args.sim_threshold,
    )
    parser: JSONActionParser = vf_env.parser  # type: ignore

    # Build OpenAI client from alias registry
    from configs.endpoints import ENDPOINTS

    ep = ENDPOINTS[args.model_alias]
    client = OpenAI(api_key=os.getenv(ep["key"], "EMPTY"), base_url=ep["url"])
    model = ep["model"]

    # Run generation
    results = vf_env.evaluate(
        client=client,
        model=model,
        num_examples=args.max_examples,
        rollouts_per_example=1,
        sampling_args={"temperature": 0.0, "response_format": {"type": "json_object"}},
    )

    # Pair truth and predictions
    paired: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for info, comp in zip(results.info, results.completion):
        gt = info if isinstance(info, dict) else {}
        ans = gt
        try:
            # info carries canonical ground truth in this env
            pred = parser.parse_answer(comp) or {}
        except Exception:
            pred = {}
        paired.append((ans, pred))

    m = metrics(paired, sim_threshold=args.sim_threshold)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    print(json.dumps(m, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

