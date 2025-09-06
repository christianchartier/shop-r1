#!/usr/bin/env python3
"""
Enhanced evaluation script that computes all three metrics from Table 2 of the Shop-R1 paper:
1. Exact Action Accuracy - All subfields must match the label
2. Action Type Accuracy - Coarse-grained classification accuracy  
3. Action Type F1 - Macro F1 across click/type_and_submit/terminate

This implements the exact evaluation criteria described in the paper:
- Click actions: Both action type and button name must match exactly
- Type_and_submit: Action type, field name, and text (via ROUGE-L >= 0.75) must match
- Terminate: Only action type needs to match
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np

try:
    from openai import OpenAI, AsyncOpenAI
except ImportError:
    print("Please install openai: pip install openai")
    sys.exit(1)

import verifiers as vf
from environments.shop_r1.shop_r1 import JSONActionParser, rouge_l


@dataclass
class EvaluationMetrics:
    """Container for all evaluation metrics matching Table 2 of the paper"""
    
    # Primary metrics (Table 2)
    exact_action_acc: float  # All subfields must match
    action_type_acc: float   # Coarse-grained action type accuracy
    action_type_f1: float    # Macro F1 across three action types
    
    # Per-class breakdown
    per_class_metrics: Dict[str, Dict[str, float]]
    
    # Additional diagnostic metrics
    total_samples: int
    confusion_matrix: Dict[str, Dict[str, int]]
    
    def to_table_row(self, model_name: str = "Model") -> str:
        """Format metrics as a row matching Table 2 format"""
        return (f"{model_name:<30} "
                f"{self.exact_action_acc:6.2%} "
                f"{self.action_type_acc:6.2%} "
                f"{self.action_type_f1:6.2%}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class PaperMetricsEvaluator:
    """Evaluator that computes metrics exactly as described in the Shop-R1 paper"""
    
    def __init__(self, sim_threshold: float = 0.75):
        """
        Initialize evaluator with paper-specified similarity threshold.
        
        Args:
            sim_threshold: ROUGE-L threshold for text similarity (paper uses 0.75)
        """
        self.sim_threshold = sim_threshold
        self.action_types = ["click", "type_and_submit", "terminate"]
    
    def canonical_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize action representation to canonical form.
        Handles various field names that might appear in different formats.
        """
        action_type = str(action.get("type", "")).lower().strip()
        
        canonical = {"type": action_type}
        
        if action_type == "terminate":
            # Terminate has no additional fields
            canonical["name"] = ""
            canonical["text"] = ""
        else:
            # Handle various field name aliases
            canonical["name"] = str(
                action.get("name") or 
                action.get("element") or 
                action.get("target") or 
                action.get("field") or
                ""
            ).strip()
            
            canonical["text"] = str(
                action.get("text") or 
                action.get("value") or 
                action.get("query") or
                action.get("input") or
                ""
            ).strip()
        
        return canonical
    
    def check_exact_match(self, truth: Dict[str, Any], pred: Dict[str, Any]) -> bool:
        """
        Check if prediction exactly matches ground truth according to paper criteria.
        
        Paper specifies:
        - Click: type and name must match exactly
        - Type_and_submit: type, name must match exactly, text via ROUGE-L >= threshold
        - Terminate: only type must match
        """
        truth = self.canonical_action(truth)
        pred = self.canonical_action(pred)
        
        # First check: action types must match
        if truth["type"] != pred["type"]:
            return False
        
        action_type = truth["type"]
        
        if action_type == "terminate":
            # For terminate, only type needs to match
            return True
            
        elif action_type == "click":
            # For click, type and name must match exactly
            return truth["name"] == pred["name"]
            
        elif action_type == "type_and_submit":
            # For type_and_submit: type and name exact, text via ROUGE-L
            name_match = truth["name"] == pred["name"]
            
            # Text similarity via ROUGE-L (as specified in paper)
            text_similarity = rouge_l(pred["text"], truth["text"])
            text_match = text_similarity >= self.sim_threshold
            
            return name_match and text_match
        
        return False
    
    def compute_metrics(self, 
                       ground_truth: List[Dict[str, Any]], 
                       predictions: List[Dict[str, Any]]) -> EvaluationMetrics:
        """
        Compute all metrics from Table 2 of the paper.
        
        Args:
            ground_truth: List of ground truth actions
            predictions: List of predicted actions
            
        Returns:
            EvaluationMetrics object with all computed metrics
        """
        assert len(ground_truth) == len(predictions), \
            f"Length mismatch: {len(ground_truth)} truth vs {len(predictions)} predictions"
        
        n = len(ground_truth)
        if n == 0:
            return EvaluationMetrics(
                exact_action_acc=0.0,
                action_type_acc=0.0,
                action_type_f1=0.0,
                per_class_metrics={},
                total_samples=0,
                confusion_matrix={}
            )
        
        # Initialize counters
        exact_matches = 0
        type_matches = 0
        
        # For per-class metrics and confusion matrix
        per_class_stats = defaultdict(lambda: {
            "total": 0,
            "exact_correct": 0,
            "type_correct": 0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0
        })
        
        confusion = defaultdict(lambda: defaultdict(int))
        
        # Process each prediction
        for truth, pred in zip(ground_truth, predictions):
            truth_canonical = self.canonical_action(truth)
            pred_canonical = self.canonical_action(pred)
            
            truth_type = truth_canonical["type"]
            pred_type = pred_canonical["type"]
            
            # Update confusion matrix
            confusion[truth_type][pred_type] += 1
            
            # Update per-class totals
            per_class_stats[truth_type]["total"] += 1
            
            # Check type match
            type_match = (truth_type == pred_type)
            if type_match:
                type_matches += 1
                per_class_stats[truth_type]["type_correct"] += 1
            
            # Check exact match
            exact_match = self.check_exact_match(truth, pred)
            if exact_match:
                exact_matches += 1
                per_class_stats[truth_type]["exact_correct"] += 1
            
            # Update for F1 calculation
            for action_type in self.action_types:
                if truth_type == action_type and pred_type == action_type:
                    per_class_stats[action_type]["true_positives"] += 1
                elif truth_type != action_type and pred_type == action_type:
                    per_class_stats[action_type]["false_positives"] += 1
                elif truth_type == action_type and pred_type != action_type:
                    per_class_stats[action_type]["false_negatives"] += 1
        
        # Calculate primary metrics
        exact_action_acc = exact_matches / n
        action_type_acc = type_matches / n
        
        # Calculate macro F1 (as specified in paper)
        f1_scores = []
        per_class_metrics = {}
        
        for action_type in self.action_types:
            stats = per_class_stats[action_type]
            
            # Precision and recall
            tp = stats["true_positives"]
            fp = stats["false_positives"]
            fn = stats["false_negatives"]
            
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            
            # F1 score
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            
            f1_scores.append(f1)
            
            # Store per-class metrics
            per_class_metrics[action_type] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": stats["total"],
                "exact_acc": stats["exact_correct"] / max(1, stats["total"]),
                "type_acc": stats["type_correct"] / max(1, stats["total"])
            }
        
        action_type_f1 = np.mean(f1_scores)
        
        return EvaluationMetrics(
            exact_action_acc=exact_action_acc,
            action_type_acc=action_type_acc,
            action_type_f1=action_type_f1,
            per_class_metrics=per_class_metrics,
            total_samples=n,
            confusion_matrix=dict(confusion)
        )
    
    def print_detailed_report(self, metrics: EvaluationMetrics):
        """Print a detailed evaluation report"""
        print("\n" + "="*70)
        print("SHOP-R1 EVALUATION REPORT (Table 2 Metrics)")
        print("="*70)
        
        # Primary metrics
        print("\nPRIMARY METRICS:")
        print("-"*40)
        print(f"Exact Action Accuracy:     {metrics.exact_action_acc:6.2%}")
        print(f"Action Type Accuracy:      {metrics.action_type_acc:6.2%}")
        print(f"Action Type F1 (Macro):    {metrics.action_type_f1:6.2%}")
        print(f"Total Samples:             {metrics.total_samples:,}")
        
        # Per-class breakdown
        print("\nPER-CLASS BREAKDOWN:")
        print("-"*40)
        print(f"{'Action Type':<20} {'Exact':<8} {'Type':<8} {'F1':<8} {'Prec':<8} {'Rec':<8} {'N':<8}")
        print("-"*70)
        
        for action_type in self.action_types:
            if action_type in metrics.per_class_metrics:
                m = metrics.per_class_metrics[action_type]
                print(f"{action_type:<20} "
                      f"{m['exact_acc']:6.2%}  "
                      f"{m['type_acc']:6.2%}  "
                      f"{m['f1']:6.2%}  "
                      f"{m['precision']:6.2%}  "
                      f"{m['recall']:6.2%}  "
                      f"{m['support']:>6}")
        
        # Confusion matrix
        print("\nCONFUSION MATRIX:")
        print("-"*40)
        header = "True\\Pred"
        print(f"{header:<20}", end="")
        for action_type in self.action_types:
            print(f"{action_type[:8]:<10}", end="")
        print()
        
        for true_type in self.action_types:
            print(f"{true_type:<20}", end="")
            for pred_type in self.action_types:
                count = metrics.confusion_matrix.get(true_type, {}).get(pred_type, 0)
                print(f"{count:<10}", end="")
            print()
        
        print("\n" + "="*70)


def load_dataset(path: str, max_examples: int = -1) -> List[Dict[str, Any]]:
    """Load JSONL dataset"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_examples > 0 and i >= max_examples:
                break
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"Warning: Failed to parse line {i+1}: {e}")
                continue
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Shop-R1 model with paper metrics (Table 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script computes the three metrics from Table 2 of the Shop-R1 paper:
  1. Exact Action Accuracy - All subfields must match
  2. Action Type Accuracy - Coarse-grained classification
  3. Action Type F1 - Macro F1 across action types

Example usage:
  python eval_paper_metrics.py --dataset data/test.jsonl --model_alias local-qwen
  python eval_paper_metrics.py --dataset data/test.jsonl --use_checkpoint checkpoints/shop_r1
        """
    )
    
    # Data arguments
    parser.add_argument("--dataset", required=True, 
                       help="JSONL file with ground truth actions")
    parser.add_argument("--max_examples", type=int, default=-1,
                       help="Maximum examples to evaluate (-1 for all)")
    
    # Model arguments
    parser.add_argument("--model_alias", default="local-qwen",
                       help="Model endpoint alias from configs/endpoints.py")
    parser.add_argument("--use_checkpoint", type=str, default=None,
                       help="Use a local checkpoint instead of API endpoint")
    
    # Evaluation arguments
    parser.add_argument("--sim_threshold", type=float, default=0.75,
                       help="ROUGE-L threshold for text similarity (paper: 0.75)")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature for generation")
    parser.add_argument("--env_strict", action="store_true",
                       help="Use strict JSON parsing")
    
    # Output arguments
    parser.add_argument("--output", default="paper_metrics.json",
                       help="Output JSON file for results")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress detailed output")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = PaperMetricsEvaluator(sim_threshold=args.sim_threshold)
    
    # Load environment
    print(f"Loading environment with dataset: {args.dataset}")
    vf_env = vf.load_environment(
        env_id="shop-r1",
        dataset_path=args.dataset,
        strict=args.env_strict,
        sim_threshold=args.sim_threshold,
    )
    parser: JSONActionParser = vf_env.parser
    
    # Setup model client
    if args.use_checkpoint:
        # TODO: Add support for loading local checkpoints
        print(f"Loading checkpoint: {args.use_checkpoint}")
        raise NotImplementedError("Checkpoint loading not yet implemented")
    else:
        from configs.endpoints import ENDPOINTS
        
        if args.model_alias not in ENDPOINTS:
            print(f"Error: Unknown model alias '{args.model_alias}'")
            print(f"Available: {', '.join(ENDPOINTS.keys())}")
            sys.exit(1)
        
        ep = ENDPOINTS[args.model_alias]
        client = OpenAI(
            api_key=os.getenv(ep["key"], "EMPTY"),
            base_url=ep["url"]
        )
        model = ep["model"]
        print(f"Using model: {model} at {ep['url']}")
    
    # Run evaluation
    print(f"Running evaluation on {args.max_examples if args.max_examples > 0 else 'all'} examples...")
    
    results = vf_env.evaluate(
        client=client,
        model=model,
        num_examples=args.max_examples,
        rollouts_per_example=1,
        sampling_args={
            "temperature": args.temperature,
            "response_format": {"type": "json_object"}
        },
    )
    
    # Extract ground truth and predictions
    ground_truth = []
    predictions = []
    
    for info, completion in zip(results.info, results.completion):
        # Ground truth
        if isinstance(info, dict):
            ground_truth.append(info)
        else:
            ground_truth.append({})
        
        # Parse prediction
        try:
            pred = parser.parse_answer(completion) or {}
        except Exception as e:
            print(f"Warning: Failed to parse prediction: {e}")
            pred = {}
        predictions.append(pred)
    
    # Compute metrics
    metrics = evaluator.compute_metrics(ground_truth, predictions)
    
    # Save results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {args.output}")
    
    # Print report
    if not args.quiet:
        evaluator.print_detailed_report(metrics)
    
    # Print Table 2 format
    print("\nTABLE 2 FORMAT:")
    print("-"*70)
    print(f"{'Model':<30} {'Exact':<8} {'Type':<8} {'F1':<8}")
    print("-"*70)
    print(metrics.to_table_row(args.model_alias or "Model"))
    print("-"*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())