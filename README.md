# <Environment Name>

> TODO: one-line description

## Environment Description

TODO

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| | | |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `done` | `bool` | Whether the episode has ended |
| `reward` | `float \| None` | Reward for the last step |

## Tasks

| Task | Difficulty | Description |
|------|------------|-------------|
| `easy_task` | Easy | TODO |
| `medium_task` | Medium | TODO |
| `hard_task` | Hard | TODO |

## Setup

### uv
```bash
uv sync
uv sync --extra dev
uv run uvicorn server.app:app --reload
```

### pip
```bash
pip install -r requirements.txt
uvicorn server.app:app --reload
```

### Docker
```bash
docker build -t my-openenv .
docker run -p 8000:8000 my-openenv
```

## Baseline Scores

| Task | Score |
|------|-------|
| `easy_task` | TBD |
| `medium_task` | TBD |
| `hard_task` | TBD |
