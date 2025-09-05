#!/usr/bin/env python3
"""Quick test script to verify Shop-R1 implementation on Prime Intellect pod."""

import json
import sys
import traceback
from pathlib import Path

# Add project root to Python path for script imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_environment():
    """Test environment loading and basic functionality."""
    print("=" * 60)
    print("SHOP-R1 QUICK TEST")
    print("=" * 60)
    
    # 1. Import test
    print("\n1. Testing imports...")
    try:
        import verifiers as vf
        print(f"✓ Verifiers version: {vf.__version__ if hasattr(vf, '__version__') else 'unknown'}")
    except ImportError as e:
        print(f"✗ Failed to import verifiers: {e}")
        return False
    
    try:
        from environments.shop_r1.shop_r1 import load_environment
        print("✓ Shop-R1 environment module imported")
    except ImportError as e:
        print(f"✗ Failed to import shop_r1: {e}")
        return False
    
    # 2. Environment loading
    print("\n2. Testing environment loading...")
    try:
        env = vf.load_environment('shop-r1')
        print(f"✓ Environment loaded via verifiers API")
        print(f"  - Parser: {env.parser.__class__.__name__}")
        # Introspect rubric using stable getters when available
        try:
            funcs = env.rubric.get_reward_funcs()
            weights = env.rubric.get_reward_weights()
        except Exception:
            funcs = getattr(env.rubric, 'funcs', [])
            weights = getattr(env.rubric, 'weights', [])
        print(f"  - Rubric functions: {len(funcs)}")
        print(f"  - Weights: {weights}")
    except Exception as e:
        print(f"✗ Failed to load environment: {e}")
        try:
            # Fallback to direct import
            env = load_environment()
            print("✓ Environment loaded via direct import (fallback)")
        except Exception as e2:
            print(f"✗ Direct import also failed: {e2}")
            return False
    
    # 3. Parser test
    print("\n3. Testing JSON parser...")
    test_cases = [
        # Valid JSON (should parse)
        ('{"rationale": "test", "action": {"type": "click", "name": "button"}}', True),
        # Invalid JSON (should not parse)
        ('not json', False),
        # Missing rationale (parse_answer may still extract action; format reward will gate strictness)
        ('{"action": {"type": "click", "name": "button"}}', True),
    ]
    
    for json_str, should_parse in test_cases:
        try:
            parsed = env.parser.parse_answer(json_str)
            if should_parse:
                if parsed is not None:
                    print(f"✓ Correctly parsed: {json_str[:50]}...")
                else:
                    print(f"✗ Failed to parse valid JSON: {json_str[:50]}...")
            else:
                if parsed is None:
                    print(f"✓ Correctly rejected invalid: {json_str[:50]}...")
                else:
                    print(f"✗ Incorrectly parsed invalid: {json_str[:50]}...")
        except Exception as e:
            print(f"✗ Parser error: {e}")
    
    # 4. Reward computation test
    print("\n4. Testing reward computation...")
    test_example = {
        "prompt": [{"role": "user", "content": "Click the search button"}],
        "answer": {"type": "click", "name": "search_button"},
        "completion": '{"rationale": "I need to search", "action": {"type": "click", "name": "search_button"}}'
    }
    
    try:
        prompt = test_example["prompt"]
        answer = test_example["answer"]
        completion = test_example["completion"]

        total_reward = 0.0
        print("  Reward components:")
        try:
            funcs = env.rubric.get_reward_funcs()
            weights = env.rubric.get_reward_weights()
        except Exception:
            funcs = getattr(env.rubric, 'funcs', [])
            weights = getattr(env.rubric, 'weights', [])
        for i, (func, weight) in enumerate(zip(funcs, weights)):
            try:
                r = func(completion, answer, prompt=prompt, info=answer)
                weighted = r * weight
                total_reward += weighted
                print(f"    Func {i}: raw={r:.3f}, weight={weight:.3f}, weighted={weighted:.3f}")
            except Exception as e:
                print(f"    Func {i}: ERROR - {e}")

        print(f"  Total reward: {total_reward:.3f}")
        if total_reward > 0:
            print("✓ Rewards computed successfully")
        else:
            print("⚠ Warning: Total reward is 0")
    except Exception as e:
        print(f"✗ Reward computation failed: {e}")
        traceback.print_exc()
    
    # 5. Dataset test
    print("\n5. Testing dataset handling...")
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    test_file = data_dir / "quick_test.jsonl"
    
    test_data = [
        {"prompt": [{"role": "user", "content": "Search"}], "answer": {"type": "type_and_submit", "name": "search", "text": "laptop"}},
        {"prompt": [{"role": "user", "content": "Click"}], "answer": {"type": "click", "name": "button"}},
        {"prompt": [{"role": "user", "content": "Done"}], "answer": {"type": "terminate"}},
    ]
    
    try:
        with open(test_file, "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")
        print(f"✓ Created test dataset: {test_file}")
        
        # Try to load it
        env2 = vf.load_environment('shop-r1', dataset_path=str(test_file))
        print(f"✓ Environment loaded with custom dataset")
    except Exception as e:
        print(f"✗ Dataset handling error: {e}")
    
    print("\n" + "=" * 60)
    print("QUICK TEST COMPLETE")
    print("=" * 60)
    return True

def test_training_scripts():
    """Test that training scripts can be imported."""
    print("\n6. Testing training script imports...")
    
    try:
        from scripts import sft_train
        print("✓ SFT script imports successfully")
    except Exception as e:
        print(f"✗ SFT script import failed: {e}")
    
    try:
        from scripts import rl_train_grpo
        print("✓ GRPO script imports successfully")
    except Exception as e:
        print(f"✗ GRPO script import failed: {e}")

if __name__ == "__main__":
    success = test_environment()
    test_training_scripts()
    
    if success:
        print("\n✅ Basic tests passed! Ready for training tests.")
        print("\nNext steps:")
        print("1. Run SFT: python scripts/sft_train.py --dataset data/quick_test.jsonl --epochs 1")
        print("2. Run GRPO: (requires vLLM server running)")
        print("3. Run eval: vf-eval shop-r1 -s -n 5")
    else:
        print("\n❌ Some tests failed. Please fix errors before proceeding.")
        sys.exit(1)
