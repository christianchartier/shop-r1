#!/usr/bin/env python3
"""
Run ablation studies from the Shop-R1 paper:
1. Model size ablations (0.5B, 1.5B, 3B)
2. Temperature sensitivity analysis
3. Different training methods comparison
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np


def run_evaluation(dataset: str, model: str, checkpoint: str = None, 
                  temperature: float = 0.0, output_file: str = None) -> Dict[str, float]:
    """Run evaluation and return metrics"""
    
    cmd = [
        "python", "scripts/eval_paper_metrics.py",
        "--dataset", dataset,
        "--temperature", str(temperature),
        "--quiet"
    ]
    
    if checkpoint:
        cmd.extend(["--use_checkpoint", checkpoint])
    else:
        cmd.extend(["--model_alias", model])
    
    if output_file:
        cmd.extend(["--output", output_file])
    else:
        output_file = f"/tmp/eval_{os.getpid()}.json"
        cmd.extend(["--output", output_file])
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        with open(output_file, "r") as f:
            results = json.load(f)
        
        return {
            "exact_action_acc": results["exact_action_acc"],
            "action_type_acc": results["action_type_acc"],
            "action_type_f1": results["action_type_f1"]
        }
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation: {e}")
        print(f"Output: {e.output}")
        return {"exact_action_acc": 0, "action_type_acc": 0, "action_type_f1": 0}


def model_size_ablation(dataset: str, output_dir: Path):
    """Table 2: Compare different model sizes"""
    
    print("\n" + "="*60)
    print("MODEL SIZE ABLATION (Table 2)")
    print("="*60)
    
    model_configs = [
        ("Qwen-2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct", [
            ("Zero-shot", None),
            ("SFT", "checkpoints/sft_0.5B"),
            ("Shop-R1", "checkpoints/shop_r1_0.5B")
        ]),
        ("Qwen-2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct", [
            ("Zero-shot", None),
            ("SFT", "checkpoints/sft_1.5B"),
            ("Shop-R1", "checkpoints/shop_r1_1.5B")
        ]),
        ("Qwen-2.5-3B", "Qwen/Qwen2.5-3B-Instruct", [
            ("Zero-shot", None),
            ("SFT", "checkpoints/sft_3B"),
            ("Shop-R1", "checkpoints/shop_r1_3B")
        ])
    ]
    
    results = {}
    
    for model_name, base_model, methods in model_configs:
        print(f"\n{model_name}")
        print("-"*40)
        
        results[model_name] = {}
        
        for method_name, checkpoint in methods:
            output_file = output_dir / f"{model_name}_{method_name.replace(' ', '_')}.json"
            
            # Skip if checkpoint doesn't exist
            if checkpoint and not Path(checkpoint).exists():
                print(f"  {method_name:<20} [Checkpoint not found]")
                continue
            
            metrics = run_evaluation(
                dataset=dataset,
                model="local-qwen",  # Will be overridden by checkpoint
                checkpoint=checkpoint,
                output_file=str(output_file)
            )
            
            results[model_name][method_name] = metrics
            
            print(f"  {method_name:<20} "
                  f"Exact: {metrics['exact_action_acc']:6.2%}  "
                  f"Type: {metrics['action_type_acc']:6.2%}  "
                  f"F1: {metrics['action_type_f1']:6.2%}")
    
    # Save results
    with open(output_dir / "model_size_ablation.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def temperature_ablation(dataset: str, checkpoint: str, output_dir: Path):
    """Figure 2: Temperature sensitivity analysis"""
    
    print("\n" + "="*60)
    print("TEMPERATURE SENSITIVITY (Figure 2)")
    print("="*60)
    
    temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = []
    
    for temp in temperatures:
        output_file = output_dir / f"temp_{temp:.1f}.json"
        
        metrics = run_evaluation(
            dataset=dataset,
            model="local-qwen",
            checkpoint=checkpoint,
            temperature=temp,
            output_file=str(output_file)
        )
        
        results.append({
            "temperature": temp,
            **metrics
        })
        
        print(f"Temperature {temp:.1f}: "
              f"Exact: {metrics['exact_action_acc']:6.2%}  "
              f"Type: {metrics['action_type_acc']:6.2%}  "
              f"F1: {metrics['action_type_f1']:6.2%}")
    
    # Save results
    with open(output_dir / "temperature_ablation.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate plot (Figure 2 style)
    if len(results) > 1:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        temps = [r["temperature"] for r in results]
        exact = [r["exact_action_acc"] for r in results]
        type_acc = [r["action_type_acc"] for r in results]
        f1 = [r["action_type_f1"] for r in results]
        
        ax1.plot(temps, exact, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel("Temperature")
        ax1.set_ylabel("Exact Action Accuracy")
        ax1.set_title("Exact Match vs Temperature")
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(temps, type_acc, 's-', linewidth=2, markersize=8, color='orange')
        ax2.set_xlabel("Temperature")
        ax2.set_ylabel("Action Type Accuracy")
        ax2.set_title("Type Accuracy vs Temperature")
        ax2.grid(True, alpha=0.3)
        
        ax3.plot(temps, f1, '^-', linewidth=2, markersize=8, color='green')
        ax3.set_xlabel("Temperature")
        ax3.set_ylabel("Macro F1 Score")
        ax3.set_title("F1 Score vs Temperature")
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "temperature_sensitivity.png", dpi=150)
        print(f"\nPlot saved to: {output_dir}/temperature_sensitivity.png")
    
    return results


def training_method_comparison(dataset: str, output_dir: Path):
    """Compare different training methods as in Table 2"""
    
    print("\n" + "="*60)
    print("TRAINING METHOD COMPARISON")
    print("="*60)
    
    methods = [
        ("Zero-shot prompting", None),
        ("RL (Binary)", "checkpoints/rl_binary"),
        ("SFT", "checkpoints/sft_shop_r1"),
        ("SFT + RL (Binary)", "checkpoints/sft_rl_binary"),
        ("Shop-R1 (Ours)", "checkpoints/rl_shop_r1")
    ]
    
    results = {}
    
    print(f"\n{'Method':<25} {'Exact':<10} {'Type':<10} {'F1':<10}")
    print("-"*55)
    
    for method_name, checkpoint in methods:
        output_file = output_dir / f"{method_name.replace(' ', '_').replace('(', '').replace(')', '')}.json"
        
        if checkpoint and not Path(checkpoint).exists():
            print(f"{method_name:<25} [Checkpoint not found]")
            continue
        
        metrics = run_evaluation(
            dataset=dataset,
            model="local-qwen",
            checkpoint=checkpoint,
            output_file=str(output_file)
        )
        
        results[method_name] = metrics
        
        print(f"{method_name:<25} "
              f"{metrics['exact_action_acc']:8.2%}  "
              f"{metrics['action_type_acc']:8.2%}  "
              f"{metrics['action_type_f1']:8.2%}")
    
    # Save results
    with open(output_dir / "training_methods.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def generate_latex_table(results: Dict[str, Any], output_file: str):
    """Generate LaTeX table for paper"""
    
    latex = r"""\begin{table}[h]
\centering
\caption{Simulation accuracy under different fine-tuning methods}
\begin{tabular}{l|ccc}
\toprule
Model & Exact Action & Action Type & Action Type \\
      & Acc. & Acc. & F1 \\
\midrule
"""
    
    for model, metrics in results.items():
        if isinstance(metrics, dict) and "exact_action_acc" in metrics:
            latex += f"{model} & "
            latex += f"{metrics['exact_action_acc']:.2%} & "
            latex += f"{metrics['action_type_acc']:.2%} & "
            latex += f"{metrics['action_type_f1']:.2%} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}"""
    
    with open(output_file, "w") as f:
        f.write(latex)
    
    print(f"\nLaTeX table saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation studies from the Shop-R1 paper",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--dataset", default="data/test.jsonl",
                       help="Test dataset path")
    parser.add_argument("--output_dir", default="results/ablations",
                       help="Output directory for results")
    
    # Ablation selection
    parser.add_argument("--model_size", action="store_true",
                       help="Run model size ablation")
    parser.add_argument("--temperature", action="store_true",
                       help="Run temperature sensitivity analysis")
    parser.add_argument("--training_methods", action="store_true",
                       help="Compare training methods")
    parser.add_argument("--all", action="store_true",
                       help="Run all ablations")
    
    # Specific configurations
    parser.add_argument("--checkpoint", type=str,
                       help="Checkpoint for temperature ablation")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check dataset exists
    if not Path(args.dataset).exists():
        print(f"Error: Dataset not found: {args.dataset}")
        sys.exit(1)
    
    results = {}
    
    # Run selected ablations
    if args.all or args.model_size:
        results["model_size"] = model_size_ablation(args.dataset, output_dir)
    
    if args.all or args.temperature:
        if not args.checkpoint:
            print("Error: --checkpoint required for temperature ablation")
            sys.exit(1)
        results["temperature"] = temperature_ablation(
            args.dataset, args.checkpoint, output_dir
        )
    
    if args.all or args.training_methods:
        results["training_methods"] = training_method_comparison(
            args.dataset, output_dir
        )
    
    # Generate LaTeX table if we have results
    if results.get("training_methods"):
        generate_latex_table(
            results["training_methods"],
            output_dir / "table2.tex"
        )
    
    print("\n" + "="*60)
    print("ABLATION STUDIES COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())