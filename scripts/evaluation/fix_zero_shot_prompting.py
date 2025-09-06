#!/usr/bin/env python3
"""
Fix for zero-shot evaluation: Add proper instruction prompting
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import argparse
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from scripts.eval_paper_metrics import PaperMetricsEvaluator

class ImprovedZeroShotEvaluator:
    """Evaluator with improved prompting for zero-shot baseline."""
    
    def __init__(self, model_alias: str = "local-qwen"):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
            base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8001/v1")
        )
        self.model = "Qwen/Qwen2.5-0.5B-Instruct"
        self.evaluator = PaperMetricsEvaluator()
    
    def create_improved_prompt(self, original_prompt: str) -> str:
        """Add clear instructions about the expected format."""
        
        instruction = """You are a web navigation assistant. You must respond with a JSON action.

IMPORTANT: You must use ONLY these action types:
- "click": Click on an element (requires name field)
- "type_and_submit": Type text and submit a form (requires name and text fields)  
- "terminate": End the task (no additional fields needed)

Response format must be EXACTLY:
{
  "rationale": "explanation of why this action",
  "type": "click|type_and_submit|terminate",
  "name": "element selector (for click/type_and_submit)",
  "text": "text to type (for type_and_submit, empty string otherwise)"
}

"""
        return instruction + "\n" + original_prompt + "\n\nJSON Action:"
    
    def get_model_response(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Get model response with improved prompting."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": self.create_improved_prompt(prompt)}
                ],
                temperature=0.1,
                max_tokens=150
            )
            
            output = response.choices[0].message.content
            
            # Parse the response
            import re
            json_matches = re.findall(r'\{[^{}]*\}', output)
            
            for match in json_matches:
                try:
                    action = json.loads(match)
                    # Validate it has the right fields
                    if 'type' in action:
                        # Ensure required fields exist
                        if 'name' not in action:
                            action['name'] = ""
                        if 'text' not in action:
                            action['text'] = ""
                        return action
                except:
                    continue
            
            return None
            
        except Exception as e:
            print(f"Error getting model response: {e}")
            return None
    
    def evaluate(self, dataset_path: str, max_examples: int = 10):
        """Run improved zero-shot evaluation."""
        
        # Load dataset
        examples = []
        with open(dataset_path, 'r') as f:
            for line in f:
                examples.append(json.loads(line))
                if len(examples) >= max_examples:
                    break
        
        print(f"\nEvaluating {len(examples)} examples with improved prompting...")
        
        results = []
        for i, example in enumerate(tqdm(examples, desc="Processing")):
            # Get the prompt
            if isinstance(example.get('prompt'), list):
                prompt = example['prompt'][0]['content']
            else:
                prompt = example.get('prompt', '')
            
            # Get expected response
            if 'response' in example:
                try:
                    expected = json.loads(example['response'])
                except:
                    expected = None
            else:
                expected = None
            
            # Get model prediction
            predicted = self.get_model_response(prompt)
            
            if i < 3:  # Show first few examples
                print(f"\n--- Example {i+1} ---")
                print(f"Prompt: {prompt[:100]}...")
                if expected:
                    print(f"Expected: type={expected.get('type')}, name={expected.get('name')}")
                if predicted:
                    print(f"Predicted: type={predicted.get('type')}, name={predicted.get('name')}")
                else:
                    print(f"Predicted: Failed to parse")
            
            results.append({
                'truth': expected,
                'prediction': predicted
            })
        
        # Calculate metrics - compute_metrics expects separate lists
        # Filter out None values (examples without ground truth)
        valid_results = [r for r in results if r['truth'] is not None]
        
        if not valid_results:
            print("\n⚠️  No examples with ground truth found in dataset!")
            print("This dataset may not have labeled responses.")
            return {
                'exact_action_acc': 0.0,
                'action_type_acc': 0.0,
                'action_type_f1': 0.0
            }
        
        print(f"\n{len(valid_results)} examples have ground truth out of {len(results)} total")
        
        truths = [r['truth'] for r in valid_results]
        predictions = [r['prediction'] for r in valid_results]
        metrics = self.evaluator.compute_metrics(truths, predictions)
        
        print("\n" + "="*60)
        print("IMPROVED ZERO-SHOT RESULTS:")
        print("="*60)
        print(f"Exact Action Accuracy:  {metrics['exact_action_acc']:.2%}")
        print(f"Action Type Accuracy:   {metrics['action_type_acc']:.2%}")
        print(f"Action Type F1 (Macro): {metrics['action_type_f1']:.2%}")
        print("="*60)
        
        return metrics

def main():
    parser = argparse.ArgumentParser(description='Improved zero-shot evaluation')
    parser.add_argument('--dataset', default='data/test.jsonl', help='Path to test dataset')
    parser.add_argument('--max_examples', type=int, default=10, help='Maximum examples to evaluate')
    
    args = parser.parse_args()
    
    evaluator = ImprovedZeroShotEvaluator()
    metrics = evaluator.evaluate(args.dataset, args.max_examples)
    
    # Save results
    output_path = 'results/evaluation/zero_shot_improved.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()