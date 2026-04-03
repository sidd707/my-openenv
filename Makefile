.PHONY: lint format check test install install-dev serve

install:
	uv sync

install-dev:
	uv sync --extra dev

lint:
	uv run ruff check .

format:
	uv run ruff format .

check: lint
	uv run ruff format --check .

test:
	uv run --extra dev python -m pytest tests/ -v --tb=short -m "not integration"

serve:
	uv run uvicorn server.app:app --reload --port 7860
