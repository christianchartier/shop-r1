#!/usr/bin/env python3
"""
Diagnostic test for evaluation pipeline to understand why metrics are 0.00%
"""

import json
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from typing import Dict, Any, Optional
import re

def parse_action(response: str) -> Optional[Dict[str, Any]]:
    """Extract JSON action from model response."""
    print(f"\n[DEBUG] Full model response:\n{response}\n")
    
    # Try to find JSON in the response
    json_matches = re.findall(r'\{[^{}]*\}', response)
    print(f"[DEBUG] Found {len(json_matches)} potential JSON objects")
    
    for i, match in enumerate(json_matches):
        print(f"[DEBUG] Attempting to parse JSON match {i+1}: {match}")
        try:
            action = json.loads(match)
            print(f"[DEBUG] Successfully parsed: {action}")
            return action
        except json.JSONDecodeError as e:
            print(f"[DEBUG] Failed to parse: {e}")
    
    # Try to extract structured response
    if "type" in response.lower():
        print("[DEBUG] Attempting to extract structured response from text")
        # Try to extract action type
        type_match = re.search(r'type["\s:]+(\w+)', response.lower())
        if type_match:
            action_type = type_match.group(1)
            print(f"[DEBUG] Found action type: {action_type}")
            
            # Build action dict
            action = {"type": action_type}
            
            # Try to extract name
            name_match = re.search(r'name["\s:]+([^",\n]+)', response, re.IGNORECASE)
            if name_match:
                action["name"] = name_match.group(1).strip()
                print(f"[DEBUG] Found name: {action['name']}")
            
            # Try to extract text
            text_match = re.search(r'text["\s:]+([^",\n]+)', response, re.IGNORECASE)
            if text_match:
                action["text"] = text_match.group(1).strip()
                print(f"[DEBUG] Found text: {action['text']}")
            
            return action
    
    print("[DEBUG] No action could be parsed from response")
    return None

def test_single_example():
    """Test evaluation on a single example with detailed debugging."""
    
    # Load a single example from test dataset
    test_file = "data/test.jsonl"
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        print("Creating a simple test example...")
        
        # Use the correct format from the Shop-R1 dataset
        example = {
            "prompt": [{"role": "user", "content": "Click on the \"Add to Cart\" button for the product \"Wireless Mouse\""}],
            "response": json.dumps({
                "rationale": "I need to click the Add to Cart button for the Wireless Mouse product",
                "type": "click", 
                "name": "button[Add to Cart]",
                "text": ""
            })
        }
        
        with open(test_file, 'w') as f:
            f.write(json.dumps(example) + "\n")
        print(f"✅ Created test file with 1 example")
    
    # Load the example
    with open(test_file, 'r') as f:
        example = json.loads(f.readline())
    
    print("\n" + "="*60)
    print("TEST EXAMPLE:")
    print("="*60)
    
    # Handle both old and new format
    if isinstance(example.get('prompt'), list):
        # New format with role/content
        prompt_text = example['prompt'][0]['content']
        print(f"Prompt: {prompt_text[:200]}...")
    else:
        prompt_text = example.get('prompt', '')
        print(f"Prompt: {prompt_text}")
    
    # Get expected action
    if 'response' in example:
        # Response is a JSON string
        try:
            expected_action = json.loads(example['response'])
            print(f"Expected Action: {json.dumps(expected_action, indent=2)}")
        except:
            print(f"Expected Response (raw): {example['response']}")
            expected_action = None
    elif 'action' in example:
        expected_action = example['action']
        print(f"Expected Action: {json.dumps(expected_action, indent=2)}")
    else:
        print("No expected action/response found in example")
        expected_action = None
    
    # Test with OpenAI API
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
        base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8001/v1")
    )
    
    print("\n" + "="*60)
    print("TESTING MODEL RESPONSE:")
    print("="*60)
    
    # Create the prompt - use the actual prompt from the example
    if isinstance(example.get('prompt'), list):
        # Use the format from the dataset
        messages = example['prompt']
        prompt = messages[0]['content'] if messages else ""
    else:
        prompt = example.get('prompt', '')
    
    print(f"Sending prompt to model...")
    print(f"Model endpoint: {client.base_url}")
    
    try:
        # For Shop-R1, the prompt already contains the full context
        # Just send it directly as the model would receive it during evaluation
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=150
        )
        
        model_output = response.choices[0].message.content
        print(f"\nModel output:\n{model_output}")
        
        # Try to parse the action
        parsed_action = parse_action(model_output)
        
        if parsed_action:
            print(f"\n✅ Successfully parsed action: {json.dumps(parsed_action, indent=2)}")
            
            # Compare with expected
            if expected_action:
                print("\n" + "="*60)
                print("COMPARISON:")
                print("="*60)
                
                # Check type match
                type_match = parsed_action.get('type') == expected_action.get('type')
                print(f"Type match: {parsed_action.get('type')} == {expected_action.get('type')} : {type_match}")
            
                # Check name match (for click and type_and_submit)
                if expected_action.get('type') in ['click', 'type_and_submit']:
                    name_match = parsed_action.get('name') == expected_action.get('name')
                    print(f"Name match: {parsed_action.get('name')} == {expected_action.get('name')} : {name_match}")
                
                # Check text match (for type_and_submit)
                if expected_action.get('type') == 'type_and_submit':
                    text_match = parsed_action.get('text') == expected_action.get('text')
                    print(f"Text match: {parsed_action.get('text')} == {expected_action.get('text')} : {text_match}")
                
                # Calculate metrics
                exact_match = parsed_action == {k: v for k, v in expected_action.items() if k != 'rationale'}
                print(f"\nExact match (ignoring rationale): {exact_match}")
            else:
                print("\n⚠️  No expected action to compare against")
            
        else:
            print("\n❌ Failed to parse action from model output")
            print("This explains why metrics are 0.00% - the model output format is not being parsed correctly")
            
    except Exception as e:
        print(f"\n❌ Error calling model: {e}")
        import traceback
        traceback.print_exc()

def test_model_understanding():
    """Test if the model understands the task format."""
    
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
        base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8001/v1")
    )
    
    print("\n" + "="*60)
    print("TESTING MODEL UNDERSTANDING:")
    print("="*60)
    
    # Simple test prompt
    test_prompts = [
        {
            "task": "Click the Submit button",
            "expected": {"type": "click", "name": "button[Submit]", "text": ""}
        },
        {
            "task": "Type 'hello world' in the search box and submit",
            "expected": {"type": "type_and_submit", "name": "input[search]", "text": "hello world"}
        },
        {
            "task": "The task is complete, terminate",
            "expected": {"type": "terminate", "name": "", "text": ""}
        }
    ]
    
    for i, test in enumerate(test_prompts, 1):
        print(f"\nTest {i}: {test['task']}")
        print(f"Expected: {test['expected']}")
        
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            messages=[
                {"role": "user", "content": f"Output a JSON action for this task: {test['task']}. Use format: {{\"type\": \"...\", \"name\": \"...\", \"text\": \"...\"}}"}
            ],
            temperature=0,
            max_tokens=100
        )
        
        output = response.choices[0].message.content
        print(f"Model output: {output}")
        
        parsed = parse_action(output)
        if parsed:
            matches = all(parsed.get(k) == v for k, v in test['expected'].items() if k in parsed)
            print(f"Parsed: {parsed}")
            print(f"Matches expected: {matches}")
        else:
            print("Failed to parse")

if __name__ == "__main__":
    print("="*60)
    print("EVALUATION PIPELINE DIAGNOSTIC TEST")
    print("="*60)
    
    # Check environment
    print("\nEnvironment:")
    print(f"OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set'}")
    print(f"OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL', 'Not set')}")
    
    # Run tests
    print("\n" + "="*60)
    print("RUNNING DIAGNOSTIC TESTS:")
    print("="*60)
    
    test_single_example()
    test_model_understanding()
    
    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)
    print("\nKey findings:")
    print("1. Check if model outputs are in the expected JSON format")
    print("2. Check if the parser can extract actions from model responses")
    print("3. Check if the model understands the task format")
    print("4. The 0.5B model may be too small to reliably output structured JSON")