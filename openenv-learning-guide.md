# OpenEnv Complete Learning Guide
## Everything You Need to Know — From Zero to Building Your Own Environment
### Based on: raun/openenv-course (Meta × HuggingFace)

---

## 1. THE BIG PICTURE — What is OpenEnv?

OpenEnv is a framework by Meta (PyTorch team) + HuggingFace that treats **RL environments as microservices**.

In traditional RL (like OpenAI Gym), your environment runs **inside your training script** — same process, same machine. This works for research but breaks in production because:

- If the environment crashes, your training crashes
- You can't scale environments independently
- You can't share environments easily
- Everything is Python-only

**OpenEnv's core idea:** Your environment runs in a **Docker container** (on HuggingFace Spaces, your machine, or the cloud), and your training code talks to it over **WebSocket/HTTP**. Just like how a frontend talks to a backend API.

```
YOUR TRAINING CODE (Python, JS, Rust — anything)
       |
       | WebSocket connection (lightweight, persistent)
       |
DOCKER CONTAINER (HuggingFace Space / Local / Cloud)
    FastAPI Server
      Your Environment Logic (reset, step, state)
    Isolated — Reproducible — Scalable
```

---

## 2. THE RL LOOP — The Foundation

Before anything else, understand this loop. Every RL system follows it:

```
while not done:
    observation = environment.observe()    # What does the world look like?
    action = agent.choose(observation)     # What should I do?
    reward = environment.step(action)      # Execute action, get feedback
    agent.learn(reward)                    # Update my strategy
```

**In plain English:**
1. Agent looks at the world (observation)
2. Agent decides what to do (action)
3. World changes based on the action and gives a score (reward)
4. Agent adjusts its strategy based on the score
5. Repeat until the task is done or time runs out

**Key terms:**
- **Episode** = one complete run from start to finish (e.g., one game of Wordle)
- **Step** = one action within an episode (e.g., one guess in Wordle)
- **Policy** = the agent's strategy for choosing actions
- **Reward signal** = the score that tells the agent how well it did

---

## 3. WHY NOT JUST USE GYM/GYMNASIUM?

| Problem | Gymnasium (Old Way) | OpenEnv (New Way) |
|---------|-------------------|-------------------|
| **Type Safety** | `obs[0][3]` — what is index 3? No idea | `obs.info_state` — IDE autocomplete, clear names |
| **Isolation** | Runs in same process — env crash = training crash | Runs in Docker container — completely isolated |
| **Deployment** | "Works on my machine" syndrome | Same Docker image everywhere |
| **Scaling** | Hard to run 1000 environments in parallel | Deploy containers, load balance with Envoy |
| **Language Lock-in** | Python only | Any language — it's just HTTP/WebSocket |
| **Sharing** | Zip files, pip install nightmares | `openenv push` → live on HuggingFace Spaces |
| **Debugging** | Cryptic numpy shape errors | Clear Pydantic validation errors with field names |

**Bottom line:** Gymnasium was built for researchers running experiments on one machine. OpenEnv is built for production — teams sharing environments, training at scale, deploying reliably.

---

## 4. THE 3-METHOD INTERFACE — The Only API You Need

Every OpenEnv environment exposes exactly 3 methods:

### `reset()` → Observation
Starts a fresh episode. Returns the initial observation (what the agent sees at the start).

```python
result = env.reset()
# result contains: observation, reward (None at start), done (False)
```

### `step(action)` → Observation
Agent takes an action. Environment executes it, updates its internal state, and returns:
- New observation (what the world looks like now)
- Reward (how good/bad was that action)
- Done flag (is the episode over?)

```python
result = env.step(MyAction(guess="crane"))
# result.observation = what you see now
# result.reward = score for this action
# result.done = True if episode is over
```

### `state()` → State
Returns episode metadata — episode ID, step count, internal state. Used for debugging and logging, not for the agent's decisions.

```python
state = env.state()
# state.episode_id, state.step_count, etc.
```

**That's it.** Every environment — whether it's Wordle, chess, file cleanup, or server migration — uses these same 3 methods. The only thing that changes is what Action, Observation, and State contain.

---

## 5. THE 3-COMPONENT PATTERN — How Every Environment is Built

Every OpenEnv environment has 3 parts:

```
my_env/
├── models.py                ← Types (Action, Observation, State)
├── client.py                ← What training code imports
├── server/
│   ├── environment.py       ← Your actual logic (reset/step/state)
│   ├── app.py               ← FastAPI server (usually 2 lines)
│   └── Dockerfile           ← Container definition
├── openenv.yaml             ← Manifest (name, version, description)
└── pyproject.toml           ← Package metadata
```

### Component 1: models.py (The Contract)

Defines the **shape** of data flowing between agent and environment using Pydantic models. These are shared between client and server — they're the contract.

```python
from openenv.core.env_server import Action, Observation, State

class MyAction(Action):
    """What the agent sends to the environment"""
    action_name: str
    parameters: dict

class MyObservation(Observation):
    """What the environment sends back"""
    # 'done' and 'reward' are inherited from base Observation
    current_state: dict
    available_actions: list[str]
    message: str

class MyState(State):
    """Episode metadata"""
    # 'episode_id' and 'step_count' are inherited from base State
    task_id: str
    total_score: float
```

**Why Pydantic?** Type validation at runtime. If the agent sends `action_name: 123` instead of a string, Pydantic throws a clear error immediately instead of failing silently deep in your logic.

### Component 2: server/environment.py (The Brain)

This is where your actual logic lives. Inherits from `Environment` base class.

```python
from openenv.core.env_server import Environment

class MyEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True  # Multiple agents at once

    def __init__(self):
        self._state = MyState()
        # Initialize your internal state here

    def reset(self, seed=None, episode_id=None, **kwargs) -> MyObservation:
        # 1. Generate fresh scenario
        # 2. Reset all internal state
        # 3. Return initial observation
        return MyObservation(done=False, reward=None, ...)

    def step(self, action: MyAction, **kwargs) -> MyObservation:
        # 1. Validate the action
        # 2. Execute it (update internal state)
        # 3. Calculate reward
        # 4. Check if episode is done
        # 5. Return new observation
        return MyObservation(done=done, reward=reward, ...)

    @property
    def state(self) -> MyState:
        return self._state
```

### Component 3: client.py (The Bridge)

What training code imports to talk to the server. Handles WebSocket connection, serialization, parsing.

```python
from openenv.core.env_client import EnvClient

class MyEnv(EnvClient[MyAction, MyObservation, MyState]):
    def _step_payload(self, action):
        """Convert action to dict for sending over WebSocket"""
        return {"action_name": action.action_name, "parameters": action.parameters}

    def _parse_result(self, payload):
        """Parse server response into StepResult"""
        obs = MyObservation(**payload.get("observation", {}))
        return StepResult(observation=obs, reward=payload.get("reward"), done=payload.get("done"))

    def _parse_state(self, payload):
        """Parse state response"""
        return MyState(**payload)
```

### The FastAPI Server (app.py — usually just 2 lines)

```python
from openenv.core.env_server import create_fastapi_app
from environment import MyEnvironment

app = create_fastapi_app(MyEnvironment)
```

`create_fastapi_app` automatically creates all the endpoints (WebSocket for reset/step/state, health check, etc.).

---

## 6. HOW THE CONNECTION WORKS — WebSocket vs HTTP

OpenEnv uses **WebSocket** (`/ws` endpoint) for the environment interaction, not regular HTTP requests.

**Why WebSocket?**
- HTTP: Each request = new TCP connection → handshake overhead (10-50ms per call)
- WebSocket: One persistent connection → each step is a tiny frame (~0.1ms overhead)
- In RL training, you call `step()` thousands of times. 10ms × 10,000 = 100 seconds wasted on HTTP. WebSocket makes this near-zero.

**How it works:**
1. Client opens a WebSocket connection to the server
2. Each `reset()`, `step()`, `state()` call sends a JSON message over this connection
3. Server processes it and sends a JSON response back
4. Connection stays open for the entire training session

**Each WebSocket connection = one isolated environment instance on the server.** So 100 connections = 100 independent games running simultaneously.

---

## 7. USING EXISTING ENVIRONMENTS — The Environment Hub

HuggingFace hosts pre-built environments. To use one:

```python
from envs.echo_env import EchoEnv

# .sync() wraps the async client for use in scripts/notebooks
with EchoEnv(base_url="https://openenv-echo-env.hf.space").sync() as env:
    result = env.reset()
    response = env.call_tool("echo_message", message="Hello!")
    print(response)  # "Hello!"
```

**Available environments include:**
- **Echo Env** — simplest possible, echoes back messages (MCP-based, uses `call_tool`)
- **OpenSpiel** — classic games: Catch, Tic-Tac-Toe, Blackjack, 2048, Cliff Walking, Kuhn Poker
- **TextArena Wordle** — word guessing game, used for LLM training in Module 5

**Two types of environments:**
1. **Standard (reset/step/state)** — most environments use this. E.g., OpenSpiel, your SafeAct env.
2. **MCP-based (tool-based)** — simpler environments use `list_tools()` and `call_tool()`. E.g., Echo Env. Think of these like function-calling — the agent calls named tools with parameters.

For our hackathon project (SafeAct), we'll use the **standard** reset/step/state pattern.

---

## 8. WRITING POLICIES — How Agents Make Decisions

A policy is just a function: observation in → action out.

**Random Policy** (baseline — usually terrible):
```python
def random_policy(observation):
    return random.choice(observation.available_actions)
```

**Heuristic Policy** (hand-coded rules — decent):
```python
def heuristic_policy(observation):
    # Apply human-written rules
    if dangerous_action_detected(observation):
        return EscalateAction(reason="Potentially irreversible")
    return safest_available_action(observation)
```

**Learned Policy** (RL-trained — what we're building the environment for):
```python
# After training, the model IS the policy
def learned_policy(observation, model):
    prompt = format_observation_as_prompt(observation)
    action = model.generate(prompt)
    return parse_action(action)
```

**Epsilon-Greedy** (exploration/exploitation balance — used during training):
```python
def epsilon_greedy(observation, model, step):
    epsilon = max(0.1, 1.0 - step / 1000)  # Start random, become strategic
    if random.random() < epsilon:
        return random_policy(observation)  # Explore
    return learned_policy(observation, model)  # Exploit
```

---

## 9. DEPLOYING YOUR ENVIRONMENT

### Option 1: Local Development (what we'll use first)

```bash
# Run the server locally
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# Test it
curl http://localhost:8000/health  # {"status": "healthy"}
```

Then connect your client to `http://localhost:8000` instead of a HF Spaces URL.

### Option 2: Docker (what judges will test)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t safeact-env .
docker run -p 8000:8000 safeact-env
```

### Option 3: HuggingFace Spaces (final deployment)

```bash
openenv push --repo-id sidd707/safeact-env
```

This gives you:
- A live API endpoint (https://sidd707-safeact-env.hf.space)
- Auto-deployed Docker container
- Web UI and API docs
- Health check endpoint

### Environment Variables for Scaling
| Variable | Default | Purpose |
|----------|---------|---------|
| `WORKERS` | 4 | Uvicorn worker processes |
| `PORT` | 8000 | Server port |
| `HOST` | 0.0.0.0 | Bind address |
| `MAX_CONCURRENT_ENVS` | 100 | Max WebSocket sessions per worker |

**HF Spaces free tier** handles ~128 concurrent sessions (2 vCPU, 16GB RAM). More than enough for judging.

---

## 10. BUILDING YOUR OWN ENVIRONMENT — Step by Step

This is the most important section for our hackathon. Here's the exact process:

### Step 1: Scaffold

```bash
openenv init safeact_env
cd safeact_env
```

This creates the full directory structure with template files.

### Step 2: Define Your Types (models.py)

Think about: What does the agent see? What can it do? What metadata do we track?

```python
from openenv.core.env_server import Action, Observation, State
from typing import Dict, List, Any, Optional

class AgentAction(Action):
    action_name: str
    parameters: Dict[str, Any]
    reasoning: str  # Agent must explain why

class SystemObservation(Observation):
    # done: bool and reward: float inherited
    task_description: str
    current_state: Dict[str, Any]
    available_actions: List[Dict[str, str]]  # name + description only
    action_history: List[str]
    steps_remaining: int
    last_action_result: str
    metadata: Dict[str, Any]

class EpisodeState(State):
    # episode_id and step_count inherited
    task_id: str
    irreversible_mistakes: int
    task_complete: bool
    total_reward: float
```

### Step 3: Write Your Logic (server/environment.py)

```python
from openenv.core.env_server import Environment

class SafeActEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        # Initialize empty — reset() sets everything up
        pass

    def reset(self, seed=None, episode_id=None, **kwargs) -> SystemObservation:
        # 1. Pick a task (easy/medium/hard)
        # 2. Generate the scenario (files, databases, services)
        # 3. Set up available actions with hidden risk labels
        # 4. Return initial observation
        ...

    def step(self, action: AgentAction, **kwargs) -> SystemObservation:
        # 1. Look up the action's hidden risk level
        # 2. Execute the action (modify internal state)
        # 3. Calculate reward (task progress + safety)
        # 4. Check if episode should end
        # 5. Return new observation
        ...

    @property
    def state(self) -> EpisodeState:
        return self._state
```

### Step 4: Wire Up FastAPI (server/app.py)

```python
from openenv.core.env_server import create_fastapi_app
from environment import SafeActEnvironment

app = create_fastapi_app(SafeActEnvironment)
```

### Step 5: Write Client (client.py)

```python
from openenv.core.env_client import EnvClient

class SafeActEnv(EnvClient[AgentAction, SystemObservation, EpisodeState]):
    def _step_payload(self, action):
        return {
            "action_name": action.action_name,
            "parameters": action.parameters,
            "reasoning": action.reasoning,
        }

    def _parse_result(self, payload):
        obs = SystemObservation(**payload.get("observation", {}))
        return StepResult(observation=obs, reward=payload.get("reward"), done=payload.get("done"))

    def _parse_state(self, payload):
        return EpisodeState(**payload)
```

### Step 6: Test Locally

```bash
uvicorn server.app:app --reload --port 8000
```

```python
from safeact_env import SafeActEnv, AgentAction

with SafeActEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset()
    print(result.observation.task_description)
    print(result.observation.available_actions)

    result = env.step(AgentAction(
        action_name="read_file",
        parameters={"path": "/data/report.txt"},
        reasoning="Need to check what files exist before deciding what to clean"
    ))
    print(f"Reward: {result.reward}, Done: {result.done}")
```

### Step 7: Docker + Deploy

```bash
docker build -t safeact-env .
docker run -p 8000:8000 safeact-env
# Test it works in Docker

openenv push --repo-id sidd707/safeact-env
```

---

## 11. THE openenv.yaml MANIFEST

Required file. Judges validate this automatically.

```yaml
name: safeact_env
version: "1.0.0"
description: "Irreversible Action Prevention — train agents to distinguish safe vs dangerous actions"
```

Keep it simple. The `openenv validate` command checks this file.

---

## 12. TRAINING WITH GRPO + TRL (Module 5 — Advanced)

This is how you actually train an LLM to play your environment. Not required for the hackathon submission, but good to understand since it's the whole point of building the environment.

### What is GRPO?

**Group Relative Policy Optimization** — a simpler alternative to PPO for training LLMs with RL.

**How it works:**
1. Give the model the same prompt multiple times
2. It generates different completions each time (because of sampling)
3. Score each completion with reward functions
4. Completions better than the group average get reinforced
5. Completions worse than the group average get suppressed

**Why GRPO over PPO?** No value model needed. Simpler to implement. Works well when you can define clear reward functions (which we can — our graders are deterministic).

### The Training Pipeline

```python
from trl import GRPOTrainer, GRPOConfig

trainer = GRPOTrainer(
    model="Qwen/Qwen3-1.7B",             # Base model to train
    reward_funcs=[                         # Multiple reward signals
        reward_task_completion,
        reward_safety_score,
        reward_escalation_quality,
    ],
    rollout_func=rollout_func,             # How model interacts with env
    train_dataset=dataset,
    args=GRPOConfig(
        learning_rate=5e-6,
        num_generations=2,                  # Completions per prompt
        gradient_accumulation_steps=64,
        max_completion_length=8,
        use_vllm=True,                      # Fast inference during training
    ),
)

trainer.train()
```

### The Rollout Function

This is the bridge between TRL and OpenEnv. It:
1. Resets the environment
2. Formats the observation as a prompt
3. Gets the model to generate an action
4. Sends the action to the environment
5. Collects rewards
6. Repeats until the episode ends

```python
def rollout_func(prompts, trainer=None):
    for prompt in prompts:
        result = env.reset()
        while not result.done:
            # Format observation → model prompt
            # Generate action with model
            # Send action to environment
            # Collect reward signals
        # Return all prompt_ids, completion_ids, logprobs, and rewards
```

### Multiple Reward Functions

Instead of one reward, you can decompose into multiple signals:
- `reward_task_completion` → Did you finish the task? (0.0 or 1.0)
- `reward_safety_score` → How safe were your choices? (0.0 to 1.0)
- `reward_escalation_quality` → Did you escalate appropriately? (0.0 to 1.0)

GRPO combines these to update the policy.

---

## 13. SCALING — How Many Sessions Can You Handle?

| Setup | Max Concurrent Sessions | Cores |
|-------|------------------------|-------|
| HF Spaces (free) | ~128 | 2 vCPU |
| Local Uvicorn | ~2,048 | 8 cores |
| Local Docker | ~2,048 | 8 cores |
| Multi-node SLURM | ~16,384 | 96 cores |

For our hackathon: HF Spaces free tier (128 sessions) is more than enough. Judges won't run 128 concurrent tests.

---

## 14. ASYNC vs SYNC

OpenEnv clients are **async by default** (for production performance):

```python
# Async (production, training scripts)
async with SafeActEnv(base_url="...") as env:
    result = await env.reset()
    result = await env.step(action)
```

```python
# Sync (notebooks, testing, quick scripts)
with SafeActEnv(base_url="...").sync() as env:
    result = env.reset()
    result = env.step(action)
```

Use `.sync()` for notebooks and testing. Use async for actual training loops.

---

## 15. QUICK REFERENCE — Commands You'll Use

```bash
# Scaffold a new environment
openenv init my_env

# Run locally
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# Health check
curl http://localhost:8000/health

# Docker
docker build -t my-env .
docker run -p 8000:8000 my-env

# Deploy to HF Spaces
openenv push --repo-id username/my-env

# Validate manifest
openenv validate

# Install from HF Space as a package
pip install git+https://huggingface.co/spaces/username/my-env
```

---

## 16. KEY TAKEAWAYS FOR OUR HACKATHON (SafeAct)

1. **We're building a microservice.** FastAPI server in a Docker container, deployed to HF Spaces.

2. **The 3 methods are all we implement:** `reset()`, `step()`, `state()`. Everything else (WebSocket handling, session management, health checks) is handled by `create_fastapi_app()`.

3. **Types matter.** Define clean Pydantic models for Action, Observation, State. This is what the judges see first and what `openenv validate` checks.

4. **The agent NEVER sees the hidden risk labels.** It only sees action names and descriptions. Learning to infer risk from context is the whole point.

5. **Reward must be partial and continuous.** Not just 0 or 1 at the end. Every step should give some signal.

6. **Graders must be deterministic.** Same episode history → same score. Always. No randomness, no LLM-as-judge.

7. **Test locally first,** then Docker, then HF Spaces. Don't skip straight to deployment.

8. **The baseline script** uses the OpenAI API client to play through all 3 tasks and report scores. It proves the environment is usable.

---

*Guide version: 1.0 | March 26, 2026*
*Source: raun/openenv-course (5 modules) + meta-pytorch/OpenEnv*
