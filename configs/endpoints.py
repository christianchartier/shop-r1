# Local endpoint registry for verifiers CLI.
#
# Use with: vf-eval shop-r1 -m local-qwen -a '{...}' -S '{...}'

ENDPOINTS = {
    "local-qwen": {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "url": "http://localhost:8000/v1",
        "key": "OPENAI_API_KEY",
    },
    # Example OpenAI entry (uncomment and set your model)
    # "openai": {
    #     "model": "gpt-4o-mini",
    #     "url": "https://api.openai.com/v1",
    #     "key": "OPENAI_API_KEY",
    # },
}

