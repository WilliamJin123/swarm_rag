# LLM-Powered Evolution System

All LLM API calls flow through `LLMClient` (client.py).

## Directory Structure

```
llm/
├── client.py               # LLMClient - SINGLE SOURCE FOR ALL API CALLS
├── factory.py              # LLMClientFactory - creates client from config
│
├── strategic_oracle.py     # Tier 1: Archive-level steering
├── tactical_advisor.py     # Tier 2: Per-genome diagnosis
├── creative_synthesizer.py # Tier 2.5: Custom expression generation
├── constrained_executor.py # Tier 3: Deterministic mutations (NO LLM)
│
├── intents.py              # MutationIntent enum + IntentAction mappings
├── evolution_journal.py    # Mutation history tracking
├── expression_builder.py   # Safe expression templates
├── decision_tracker.py     # Agent decision tracking for behavioral analysis
├── parsers.py              # Expression string parsing
├── utils.py                # Genome-to-context conversion
│
└── __init__.py             # Public exports
```

## Where LLM Calls Happen

**Single entry point:** `LLMClient.call()` in `client.py:76-109`

| Component | Method | Calls |
|-----------|--------|-------|
| Strategic Oracle | `get_directive()` | `self.client.call(system_prompt, user_prompt)` |
| Tactical Advisor | `get_prescription()` | `self.client.call(system_prompt, user_prompt)` |
| Creative Synthesizer | `synthesize()` | `self.client.call(system_prompt, user_prompt)` |
| Constrained Executor | `execute()` | **NO LLM CALLS** (deterministic) |

## LLMClient

```python
# client.py - Single source of truth for all LLM API calls

class LLMClient:
    def __init__(self, wrapper, model, max_retries=3, retry_delay=1.0):
        self.wrapper = wrapper  # keycycle.MultiProviderWrapper
        self.model = model

    def call(self, system_prompt, user_prompt, json_mode=True, temperature=0.3):
        """All LLM calls go through here."""
        # Retry with exponential backoff
        for attempt in range(self.max_retries):
            try:
                client = self.wrapper.get_openai_client()
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"} if json_mode else None,
                    temperature=temperature,
                )
                return LLMCallResult(content=..., parsed=..., success=True)
            except Exception as e:
                # Exponential backoff retry
                ...
        return LLMCallResult(success=False, error=...)

    @classmethod
    def from_config(cls, provider, model, env_path):
        """Create from config (loads API keys from .env)"""
        ...
```

## Three-Tier Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TIER 1: STRATEGIC ORACLE                  │
│         strategic_oracle.py - get_directive()               │
│                                                              │
│  When: Every N generations OR stagnation detected           │
│  Input: Archive stats, QD trends, success rates             │
│  Output: EvolutionMode, FocusComponent, Temperature         │
│  LLM: self.client.call(system_prompt, user_prompt)          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   TIER 2: TACTICAL ADVISOR                   │
│         tactical_advisor.py - get_prescription()            │
│                                                              │
│  When: Per-genome mutation                                  │
│  Input: Genome metrics, behavioral signature, directive     │
│  Output: Diagnosis, MutationIntent, Confidence              │
│  LLM: self.client.call(system_prompt, user_prompt)          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              TIER 2.5: CREATIVE SYNTHESIZER (optional)       │
│         creative_synthesizer.py - synthesize()              │
│                                                              │
│  When: Stagnation, low coverage, periodic experimentation   │
│  Input: Genome, prescription, archive state                 │
│  Output: Custom expressions (AST-validated)                 │
│  LLM: self.client.call(system_prompt, user_prompt)          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 TIER 3: CONSTRAINED EXECUTOR                 │
│         constrained_executor.py - execute()                 │
│                                                              │
│  *** NO LLM CALLS - Pure deterministic logic ***            │
│  Input: Genome, MutationPrescription, Directive             │
│  Output: Validated parameter + expression changes           │
└─────────────────────────────────────────────────────────────┘
```

## Usage

### Creating an LLMClient

```python
# From config
from swarm_rag.evolution.llm import LLMClientFactory
client = LLMClientFactory.create(config)

# Or directly
from swarm_rag.evolution.llm import LLMClient
client = LLMClient.from_config(
    provider="cerebras",
    model="zai-glm-4.7",
    env_path=".env"
)
```

### Using ThreeTierMutator

```python
from swarm_rag.evolution.llm import LLMClient, ThreeTierMutator

client = LLMClient.from_config(provider="cerebras", model="zai-glm-4.7")
mutator = ThreeTierMutator(client, bounds=ParameterBounds())

# Mutate a genome
result, prescription = mutator.mutate(genome, journal)
```

## Provider Setup

```env
# .env file
CEREBRAS_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

Supported: `cerebras`, `openai`, `groq`, `anthropic`, `together`, `fireworks`, `deepseek`

## Debugging LLM Calls

All LLM calls go through `LLMClient.call()`. To debug:

1. Enable logging: `logging.getLogger("swarm_rag.evolution.llm.client").setLevel(logging.DEBUG)`
2. Check `LLMCallResult.error` for failure details
3. Check `LLMCallResult.attempts` to see retry count

## Data Flow

```
Generation N starts
    │
    ├─► LLMClientFactory.create(config) → LLMClient
    │
    ├─► ThreeTierMutator(client)
    │       │
    │       ├─► StrategicOracle(client)
    │       ├─► TacticalAdvisor(client)
    │       └─► CreativeSynthesizer(client)  # optional
    │
    ├─► oracle.get_directive() → client.call() → StrategicDirective
    │
    ├─► For each genome:
    │       ├─► advisor.get_prescription() → client.call() → MutationPrescription
    │       ├─► [optional] synthesizer.synthesize() → client.call() → CreativeProposal
    │       └─► executor.execute() → (NO LLM) → ExecutionResult
    │
    └─► Journal records outcomes
```

## Key Types

```python
@dataclass
class LLMCallResult:
    content: str                    # Raw response
    parsed: Optional[Dict]          # JSON parsed (if json_mode)
    success: bool
    error: Optional[str]
    attempts: int                   # How many retries

class MutationIntent(Enum):
    INCREASE_EXPLORATION = "increase_exploration"
    REDUCE_LOOPS = "reduce_loops"
    IMPROVE_COVERAGE = "improve_coverage"
    # ... 17 total

@dataclass
class StrategicDirective:
    mode: EvolutionMode             # explore_params, exploit_top, etc.
    focus_component: TargetComponent
    exploration_temperature: float

@dataclass
class MutationPrescription:
    diagnosis: str
    primary_intent: MutationIntent
    target_component: TargetComponent
    confidence: float
```
