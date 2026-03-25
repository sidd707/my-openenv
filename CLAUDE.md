# CLAUDE.md — Project Memory & Guidelines
# Meta × HuggingFace OpenEnv Hackathon
# Team: Peaky Blinders | Siddharth Patel + Sarthak Chauhan

---

## ⚠️ CRITICAL RULES — READ BEFORE DOING ANYTHING

1. **NEVER start editing/writing code without explicit permission.**
   - Always PLAN first. Present the plan. Wait for approval. Then execute.
   - If you find a bug or issue, TELL ME first. Don't silently fix it.

2. **NEVER assume. Always ask.**
   - If something is unclear, stop and ask before proceeding.
   - One question at a time — don't bombard with multiple questions.

3. **ALWAYS check root cause before suggesting a fix.**
   - Don't treat symptoms. Understand WHY something is broken first.

4. **Session End Protocol (IMPORTANT):**
   - When I say "closing for today" or "session end" — update the
     SESSION PROGRESS LOG section at the bottom of this file.
   - Write: what was completed, what is in progress, what is next.
   - This is how you remember context across sessions.

---

## 🎯 WHAT WE ARE BUILDING

### The Hackathon
- **Organizer:** Scaler School of Technology
- **Partners:** Meta (PyTorch team) + HuggingFace
- **Team Name:** Peaky Blinders
- **Team Lead:** Siddharth Patel (sidd707888@gmail.com)
- **Teammate:** Sarthak Chauhan (sarthak4156@gmail.com)
- **Submission Window:** March 28 – April 7, 2026 (11:59 PM deadline)
- **Results:** April 10, 2026

### The Task (Plain English)
Build a real-world AI training environment using the OpenEnv framework.
An AI agent should be able to "learn" by interacting with our environment
through a standard API: step() / reset() / state().

Think of it like this: we are building a "gym" but instead of games,
our gym simulates something humans do in real work — like email triage,
customer support, code review, data cleaning, etc.

---

## 📋 PROBLEM STATEMENT (Detailed)

### Core Requirement
Build a complete OpenEnv environment that:
- Simulates a REAL-WORLD task (NOT a game, NOT a toy problem)
- Implements the full OpenEnv spec
- Has minimum 3 tasks (easy → medium → hard)
- Has a meaningful reward function (partial progress, not just win/lose)
- Has a baseline inference script using OpenAI API
- Deploys to HuggingFace Spaces with Docker

### The OpenEnv API (This is what we MUST implement)

```python
reset()         # Start a fresh episode → returns initial Observation
step(action)    # Agent takes an action → returns Observation (with reward)
state()         # Returns current episode metadata
```

### Typed Models (Pydantic — MANDATORY)
We must define these as proper Pydantic models:
- `Observation` — what the agent sees (must include `reward`, `done`, `metadata`)
- `Action` — what the agent can do
- `State` — episode metadata (step count, episode id, etc.)

### Required Files (Every single one is mandatory)
| File | Purpose |
|------|---------|
| `openenv.yaml` | Environment metadata — validated automatically by judges |
| `Dockerfile` | Must build and run cleanly with `docker build && docker run` |
| `README.md` | Environment description, action/obs spaces, setup, baseline scores |
| `baseline.py` | Inference script using OpenAI API client + OPENAI_API_KEY env var |

### Required HTTP Endpoints (Beyond the core API)
| Endpoint | What it does |
|----------|-------------|
| `/baseline` | Triggers inference and returns baseline scores for all 3 tasks |
| `/grader` | Returns grader score after an episode is completed |
| `/tasks` | Returns list of tasks + action schema for each |

### The 3 Tasks (Minimum)
- **Task 1 (Easy):** Simple, baseline agents should score reasonably well
- **Task 2 (Medium):** Requires some reasoning
- **Task 3 (Hard):** Should genuinely challenge frontier models like GPT-4

Each task needs:
- A clear objective
- A programmatic grader that scores 0.0 to 1.0
- Deterministic and reproducible scoring
- Partial reward signals along the way (not just final binary score)

### Reward Function Rules
- Must provide signal throughout the full trajectory
- Must reward partial progress (e.g., agent got halfway = 0.5, not 0.0)
- Must penalize clearly bad behavior (infinite loops, destructive actions)
- NOT allowed to always return the same score (instant disqualification)

---

## 🏆 SCORING CRITERIA (How judges evaluate us)

| Criteria | Weight | What it means |
|----------|--------|---------------|
| Real-world utility | 30% | Would an RL researcher actually use this? Does it fill a real gap? |
| Task & grader quality | 25% | Are tasks well-defined? Are graders deterministic? Does hard task challenge GPT-4? |
| Environment design | 20% | Clean state management, good reward shaping, proper episode boundaries |
| Code quality & spec compliance | 15% | openenv validate passes, Docker builds, HF Space deploys |
| Creativity & novelty | 10% | Domain not seen in OpenEnv before, clever reward design |

### Scoring to aim for (Real-world utility):
- 0-5: Toy problem → AVOID AT ALL COSTS
- 6-15: Valid but shallow → NOT GOOD ENOUGH
- 16-25: Good domain modeling → Acceptable
- **26-30: Fills a real gap, immediate value for RL community → TARGET THIS**

---

## 🚫 INSTANT DISQUALIFICATION (Never do these)

1. HF Space doesn't deploy or respond to `reset()`
2. Graders that always return the same score
3. No baseline inference script
4. Plagiarized or copied environment
5. `openenv validate` fails

---

## 🔧 TECH STACK

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Main language |
| FastAPI | HTTP server for the environment |
| Pydantic v2 | Typed models for Action, Observation, State |
| uvicorn | ASGI server to run FastAPI |
| openenv-core | OpenEnv framework library |
| Docker | Containerization |
| HuggingFace Spaces | Deployment target |
| OpenAI Python client | Baseline inference script |
| uv | Package manager (use this, NOT pip directly) |

### Package Manager Rules
- Always use `uv pip install` not `pip install`
- Virtual env is at `.venv/` — always activate before running anything
- Dependencies go in `requirements.txt`

### Code Style Rules
- Always use Pydantic v2 syntax (not v1)
- Type hints everywhere — no untyped functions
- FastAPI endpoints must have proper response_model declarations
- Environment class must inherit from OpenEnv base class
- All grader functions must be deterministic (same input = same output always)

---

## 📁 EXPECTED PROJECT STRUCTURE

```
Meta-hugginface-openenv-/
├── CLAUDE.md                  ← This file (project memory)
├── README.md                  ← For judges and HuggingFace
├── openenv.yaml               ← OpenEnv metadata (validated by judges)
├── Dockerfile                 ← Must build and run cleanly
├── requirements.txt           ← All dependencies
├── .env.example               ← Template (never commit real .env)
│
├── app/
│   ├── main.py               ← FastAPI app entry point
│   ├── environment.py        ← Core environment logic (reset/step/state)
│   ├── models.py             ← Pydantic models (Action, Observation, State)
│   ├── tasks/
│   │   ├── task_easy.py
│   │   ├── task_medium.py
│   │   └── task_hard.py
│   └── graders/
│       ├── grader_easy.py
│       ├── grader_medium.py
│       └── grader_hard.py
│
├── baseline.py               ← Inference script (uses OPENAI_API_KEY)
└── tests/
    └── test_environment.py
```

---

## 🔑 KEY COMMANDS TO REMEMBER

```bash
# Activate virtual environment
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows

# Install dependencies
uv pip install -r requirements.txt

# Run the server locally
uvicorn app.main:app --reload --port 8000

# Test the environment manually
curl http://localhost:8000/reset
curl -X POST http://localhost:8000/step -H "Content-Type: application/json" -d '{"action": "..."}'
curl http://localhost:8000/state
curl http://localhost:8000/tasks
curl http://localhost:8000/grader
curl http://localhost:8000/baseline

# Docker commands
docker build -t my-openenv .
docker run -p 8000:8000 my-openenv

# Run baseline script
OPENAI_API_KEY=your_key python baseline.py

# Validate OpenEnv spec
openenv validate
```

---

## 🧑‍💻 HOW WE WORK (WORKFLOW)

### Every coding session follows this order:
1. Read this CLAUDE.md first to understand current state
2. Check SESSION PROGRESS LOG to know where we left off
3. Discuss the next task with Siddharth
4. **PLAN** → present the plan → get approval → **THEN** code
5. After completing something, update the progress log

### For every new feature/change:
- Step 1: Explain what you want to do and WHY
- Step 2: Show the plan (files affected, approach)
- Step 3: Wait for "go ahead" from Siddharth
- Step 4: Implement
- Step 5: Explain what was done

### When you find a bug:
- DO NOT silently fix it
- Say: "I found an issue in [file]. The problem is [X]. Here's my proposed fix: [Y]. Should I proceed?"

---

## 📌 DOMAIN DECISION

**[TO BE FILLED — pending domain discussion with mentor]**

Current status: Domain not yet decided. This section will be updated
once we finalize what real-world task our environment simulates.

Leading options being considered: TBD

---

## 📊 SESSION PROGRESS LOG

> This section is updated by Claude at the END of every session.
> Format: Date | What was done | What is in progress | What is next

---

### Session 1 — [March 25, 2026]
**Status:** Setup & Planning phase

**Completed:**
- Set up project folder structure (META-HACKATHON)
- Created virtual environment (.venv)
- Cloned project repository
- Created CLAUDE.md (this file)
- Understood full problem statement and judging criteria
- Learned about OpenEnv framework (Meta + HuggingFace, launched Oct 2025)
- Understood Claude Code setup strategy (CLAUDE.md, rules, minimal skills)

**In Progress:**
- Domain selection (the most important decision before writing any code)

**Next Session Must Do:**
- Finalize the domain/use-case for our environment
- Install Docker and verify it works
- Study Module 4 of the prep course (Building Your Own Environment)
- Set up basic FastAPI skeleton once domain is locked

**Blockers:**
- None currently

---
<!-- Future sessions will be appended below this line -->