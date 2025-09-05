import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

try:
    from torch.utils.data import Dataset as TorchDataset
except Exception:
    class TorchDataset(object):
        pass
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    default_data_collator,
)

import verifiers as vf


def _ensure_chat_list(x: Any) -> List[Dict[str, str]]:
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return [{"role": "user", "content": x}]
    return []


def _canonical_action(ans: Dict[str, Any]) -> Dict[str, Any]:
    t = str(ans.get("type", "")).lower()
    out: Dict[str, Any] = {"type": t}
    if t == "terminate":
        out["name"] = ""
        out["text"] = ""
    else:
        name = ans.get("name") or ans.get("element") or ans.get("target") or ""
        text = ans.get("text") or ans.get("value") or ""
        out["name"] = str(name)
        out["text"] = str(text)
    return out


def _build_assistant_json(rationale: Optional[str], action: Dict[str, Any]) -> str:
    rat = rationale or ""
    obj = {"rationale": rat, "action": _canonical_action(action)}
    return json.dumps(obj, ensure_ascii=False)


class SFTJsonlDataset(TorchDataset):
    def __init__(self, path: str, tokenizer, max_seq_len: int = 32768) -> None:
        self.rows: List[Tuple[List[Dict[str, str]], str]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                prompt = _ensure_chat_list(obj.get("prompt"))
                rat = obj.get("rationale") if isinstance(obj.get("rationale"), str) else None
                ans = obj.get("answer") or obj.get("action") or {}
                if not isinstance(ans, dict):
                    continue
                assistant = _build_assistant_json(rat, ans)
                self.rows.append((prompt, assistant))
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        import torch  # lazy import to allow --help without torch installed
        prompt, assistant_text = self.rows[idx]
        # Build prompt-only and full chat sequences
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )
        chat = list(prompt) + [{"role": "assistant", "content": assistant_text}]
        full_ids = self.tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )
        input_ids = full_ids[0]
        labels = input_ids.clone()
        # Mask prompt tokens
        prompt_len = prompt_ids.shape[-1]
        labels[:prompt_len] = -100
        # Truncate to max_seq_len from the left (long contexts)
        if input_ids.shape[0] > self.max_seq_len:
            input_ids = input_ids[-self.max_seq_len :]
            labels = labels[-self.max_seq_len :]
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def main():
    # Import torch lazily so that --help works without heavy deps
    import torch  # noqa: F401
    ap = argparse.ArgumentParser(description="Shop-R1 SFT training (rationale + action)")
    ap.add_argument("--dataset", required=True, help="SFT JSONL path with prompt/rationale/answer")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="HF model id or path")
    ap.add_argument("--output_dir", default="checkpoints/sft", help="Output directory")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--per_device_batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=64, help="Gradient accumulation steps")
    ap.add_argument("--max_seq_len", type=int, default=32768)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    # Load model/tokenizer (fallback to transformers if verifiers API is absent)
    # TEMPORARY FIX: Force fallback to avoid FlashAttention2 requirement
    get_mat = None  # Force use of direct transformers loading with SDPA
    if False:  # Disable verifiers loading temporarily
        model, tokenizer = get_mat(args.model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype="auto",
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
        # Disable cache during training for memory/runtime stability
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    ds = SFTJsonlDataset(args.dataset, tokenizer, max_seq_len=args.max_seq_len)
    # Use default_data_collator so that labels/attention_mask are padded consistently
    collator = default_data_collator

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        bf16=args.bf16,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to=["none"],
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
