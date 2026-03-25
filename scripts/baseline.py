"""
Baseline inference script.
Uses the OpenAI client to run one episode per task against the environment.
Reads OPENAI_API_KEY from environment variables.
Prints scores as JSON to stdout.
Usage: OPENAI_API_KEY=sk-... python scripts/baseline.py
"""
