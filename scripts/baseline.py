"""
Convenience wrapper — delegates to root baseline.py.
Usage: python scripts/baseline.py [--task easy] [--url ws://localhost:8000/ws]
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline import main, argparse, asyncio  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SafeAct-Env baseline agent")
    parser.add_argument("--url", default="ws://localhost:8000/ws", help="WebSocket URL")
    parser.add_argument("--task", default=None, help="Single task to run (easy/medium/hard)")
    args = parser.parse_args()
    asyncio.run(main(args))
