# Local endpoint registry for verifiers CLI.
#
# Use with: vf-eval shop-r1 -m local-qwen -a '{...}' -S '{...}'

import os

ENDPOINTS = {
    "local-qwen": {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",  # Default to 0.5B for evaluation
        "url": os.getenv("OPENAI_BASE_URL", "http://localhost:8001/v1"),  # Use env var or default to 8001
        "key": "OPENAI_API_KEY",
    },
    "local-qwen-3b": {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "url": os.getenv("OPENAI_BASE_URL", "http://localhost:8001/v1"),
        "key": "OPENAI_API_KEY",
    },
    "local-qwen-1.5b": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "url": os.getenv("OPENAI_BASE_URL", "http://localhost:8001/v1"),
        "key": "OPENAI_API_KEY",
    },
    # Training server (port 8000)
    "vllm-train": {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "url": "http://localhost:8000/v1",
        "key": "OPENAI_API_KEY",
    },
    # Evaluation server (port 8001)
    "vllm-eval": {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "url": "http://localhost:8001/v1",
        "key": "OPENAI_API_KEY",
    },
    # Example OpenAI entry (uncomment and set your model)
    # "openai": {
    #     "model": "gpt-4o-mini",
    #     "url": "https://api.openai.com/v1",
    #     "key": "OPENAI_API_KEY",
    # },
}

