#!/usr/bin/env python3
"""Check the format of the test dataset."""

import json
import sys

def check_dataset(file_path="data/test.jsonl"):
    """Check the format of examples in the dataset."""
    
    print(f"Checking dataset: {file_path}\n")
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    print(f"Total examples: {len(lines)}")
    
    # Check first few examples
    for i, line in enumerate(lines[:3]):
        example = json.loads(line)
        print(f"\n--- Example {i+1} ---")
        
        # Check what fields are present
        print(f"Fields: {list(example.keys())}")
        
        # Check prompt format
        if 'prompt' in example:
            if isinstance(example['prompt'], list):
                print(f"Prompt type: list of {len(example['prompt'])} messages")
                if example['prompt']:
                    print(f"First message role: {example['prompt'][0].get('role', 'N/A')}")
                    content = example['prompt'][0].get('content', '')
                    print(f"Content preview: {content[:100]}...")
            else:
                print(f"Prompt type: {type(example['prompt'])}")
        
        # Check response format
        if 'response' in example:
            resp = example['response']
            print(f"Response type: {type(resp)}")
            if isinstance(resp, str):
                # Try to parse as JSON
                try:
                    parsed = json.loads(resp)
                    print(f"Response is JSON string with fields: {list(parsed.keys())}")
                    if 'type' in parsed:
                        print(f"Action type: {parsed['type']}")
                except:
                    print(f"Response preview: {resp[:100]}...")
        else:
            print("No 'response' field")
        
        # Check for action field
        if 'action' in example:
            print(f"Has 'action' field: {example['action']}")
    
    # Check how many have responses
    with_response = sum(1 for line in lines if 'response' in json.loads(line))
    print(f"\n{with_response}/{len(lines)} examples have 'response' field")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_dataset(sys.argv[1])
    else:
        check_dataset()