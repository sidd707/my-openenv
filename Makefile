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
	uv run pytest

serve:
	uv run uvicorn server.app:app --reload
