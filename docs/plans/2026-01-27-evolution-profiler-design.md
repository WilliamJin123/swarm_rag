# Evolution Loop Profiler Design

**Date:** 2026-01-27
**Goal:** Add visibility into evolution loop performance bottlenecks (wall-clock time + GPU memory)

## Problem

The evolution loop lacks visibility into where time and memory are spent. Current state:
- `StepProfiler` exists for retrieval hot paths but only covers swarm traversal
- `MemoryProfiler` exists but isn't integrated into the evolution loop
- Scattered `time.time()` calls in evaluator with no unified reporting

We need generation-level profiling to identify:
1. Which phase dominates wall-clock time (breeding vs evaluation vs archive)
2. Memory allocation patterns per phase
3. Potential memory leaks across generations

## Design

### GenerationProfiler Class

**Location:** `swarm_rag/evolution/execution/profiler.py`

**Activation:** Environment variable `EVOLUTION_PROFILE=1` (matches `StepProfiler` pattern)

```python
import os
import time
import torch
from typing import Dict, Tuple, List
from contextlib import contextmanager


class GenerationProfiler:
    """
    Generation-level profiler with GPU memory tracking.

    Enable with EVOLUTION_PROFILE=1 environment variable.

    Usage:
        profiler = GenerationProfiler.from_env()

        for gen in range(n_generations):
            profiler.start_generation(gen)

            with profiler.section("breeding"):
                offspring = breed()

            with profiler.section("evaluation"):
                evaluate(offspring)

            # Print live stats for this generation
            if profiler.enabled:
                logger.info(profiler.end_generation())

        # Print final summary
        print(profiler.summary())
    """

    __slots__ = (
        'enabled', 'generation_timings', 'generation_memory',
        'current_gen', '_has_cuda', '_gen_start_mem'
    )

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.generation_timings: Dict[int, Dict[str, float]] = {}
        self.generation_memory: Dict[int, Dict[str, Tuple[int, int]]] = {}  # (before, after) bytes
        self.current_gen: int = 0
        self._has_cuda = torch.cuda.is_available()
        self._gen_start_mem: int = 0

    @classmethod
    def from_env(cls) -> "GenerationProfiler":
        """Create profiler from environment variable."""
        enabled = os.environ.get('EVOLUTION_PROFILE', '0') == '1'
        return cls(enabled=enabled)

    def _get_gpu_mem(self) -> int:
        """Get current GPU memory allocated in bytes."""
        if self._has_cuda:
            return torch.cuda.memory_allocated()
        return 0

    def start_generation(self, gen: int):
        """Begin profiling a new generation."""
        self.current_gen = gen
        self.generation_timings[gen] = {}
        self.generation_memory[gen] = {}
        self._gen_start_mem = self._get_gpu_mem()

    @contextmanager
    def section(self, name: str):
        """Profile a named section within the current generation."""
        if not self.enabled:
            yield
            return

        mem_before = self._get_gpu_mem()
        start = time.perf_counter()
        yield
        elapsed_ms = (time.perf_counter() - start) * 1000
        mem_after = self._get_gpu_mem()

        self.generation_timings[self.current_gen][name] = elapsed_ms
        self.generation_memory[self.current_gen][name] = (mem_before, mem_after)

    def end_generation(self) -> str:
        """
        Returns live summary string for this generation.

        Example output:
            [1247ms | GPU:2841MB] breed=42ms | eval=1180ms(+12.3MB) | archive=12ms(-11.8MB)
        """
        if not self.enabled:
            return ""

        timings = self.generation_timings.get(self.current_gen, {})
        memory = self.generation_memory.get(self.current_gen, {})
        total_ms = sum(timings.values())

        parts = []
        for name, ms in timings.items():
            mem_before, mem_after = memory.get(name, (0, 0))
            delta_mb = (mem_after - mem_before) / 1024 / 1024
            if abs(delta_mb) > 0.1:  # Only show if >0.1MB change
                parts.append(f"{name}={ms:.0f}ms({delta_mb:+.1f}MB)")
            else:
                parts.append(f"{name}={ms:.0f}ms")

        current_mb = self._get_gpu_mem() / 1024 / 1024
        return f"[{total_ms:.0f}ms | GPU:{current_mb:.0f}MB] " + " | ".join(parts)

    def summary(self) -> str:
        """
        Final summary across all generations.

        Example output:
            === Evolution Profile Summary (100 generations) ===
            Phase           Time (s)    %      Avg/gen    Mem Δ (MB)
            ────────────────────────────────────────────────────────
            evaluation      118.2s    94.7%    1182ms     +12.3 → -12.1
            breeding          4.1s     3.3%      41ms     +0.2 → -0.1
            archive           1.2s     1.0%      12ms     -11.8 → +0.0
            ────────────────────────────────────────────────────────
            TOTAL           124.8s              1248ms

            Memory trend: Gen 0: 2840MB → Gen 99: 2842MB (+2MB)
        """
        if not self.enabled or not self.generation_timings:
            return "Profiling disabled or no data collected"

        n_gens = len(self.generation_timings)

        # Aggregate timings per section
        section_totals: Dict[str, float] = {}
        section_mem_deltas: Dict[str, List[float]] = {}

        for gen, timings in self.generation_timings.items():
            for name, ms in timings.items():
                section_totals[name] = section_totals.get(name, 0) + ms

                if gen in self.generation_memory and name in self.generation_memory[gen]:
                    before, after = self.generation_memory[gen][name]
                    delta_mb = (after - before) / 1024 / 1024
                    if name not in section_mem_deltas:
                        section_mem_deltas[name] = []
                    section_mem_deltas[name].append(delta_mb)

        total_ms = sum(section_totals.values())
        total_s = total_ms / 1000

        lines = [
            f"=== Evolution Profile Summary ({n_gens} generations) ===",
            f"{'Phase':<16} {'Time (s)':<12} {'%':<8} {'Avg/gen':<12} {'Mem Δ (MB)':<16}",
            "─" * 70,
        ]

        # Sort by time descending
        sorted_sections = sorted(section_totals.items(), key=lambda x: -x[1])

        for name, total_section_ms in sorted_sections:
            pct = (total_section_ms / total_ms * 100) if total_ms > 0 else 0
            avg_ms = total_section_ms / n_gens

            # Memory delta stats
            mem_deltas = section_mem_deltas.get(name, [])
            if mem_deltas:
                avg_delta = sum(mem_deltas) / len(mem_deltas)
                mem_str = f"{avg_delta:+.1f} avg"
            else:
                mem_str = "N/A"

            lines.append(
                f"{name:<16} {total_section_ms/1000:>8.1f}s    {pct:>5.1f}%    {avg_ms:>8.0f}ms    {mem_str:<16}"
            )

        lines.append("─" * 70)
        lines.append(
            f"{'TOTAL':<16} {total_s:>8.1f}s             {total_ms/n_gens:>8.0f}ms"
        )

        # Memory trend across generations
        gens_sorted = sorted(self.generation_timings.keys())
        if len(gens_sorted) >= 2:
            first_gen = gens_sorted[0]
            last_gen = gens_sorted[-1]

            # Get memory at start of first gen and end of last gen
            first_mem = self.generation_memory.get(first_gen, {})
            last_mem = self.generation_memory.get(last_gen, {})

            if first_mem and last_mem:
                # Use first section's "before" as gen start
                first_section = list(first_mem.keys())[0] if first_mem else None
                last_section = list(last_mem.keys())[-1] if last_mem else None

                if first_section and last_section:
                    start_mb = first_mem[first_section][0] / 1024 / 1024
                    end_mb = last_mem[last_section][1] / 1024 / 1024
                    drift = end_mb - start_mb

                    leak_status = "No leak detected" if abs(drift) < 10 else "POTENTIAL LEAK"
                    lines.append("")
                    lines.append(
                        f"Memory trend: Gen {first_gen}: {start_mb:.0f}MB → "
                        f"Gen {last_gen}: {end_mb:.0f}MB ({drift:+.0f}MB) "
                        f"{'✓ ' + leak_status if abs(drift) < 10 else '⚠ ' + leak_status}"
                    )

        return "\n".join(lines)
```

### Integration into MAP-Elites Loop

**File:** `swarm_rag/evolution/orchestrators/map_elites.py`

```python
# In MAPElitesOrchestrator.__init__():
from ..execution.profiler import GenerationProfiler
self._profiler = GenerationProfiler.from_env()

# In optimize() method, wrap each phase:
for gen in pbar:
    self._profiler.start_generation(gen)

    with self._profiler.section("strategic_oracle"):
        self.me_loop.update_strategic_directive(self.archive)

    with self._profiler.section("breeding"):
        offspring = self.me_loop.step(self.archive)

    if not offspring:
        break

    with self._profiler.section("evaluation"):
        self.evaluator.evaluate(offspring)

    with self._profiler.section("fitness_assign"):
        self.fitness_strategy.assign_fitness(offspring, generation=gen)

    with self._profiler.section("archive_insert"):
        added_count = 0
        for child in offspring:
            added = self.archive.add(child)
            # ... rest of insertion logic

    with self._profiler.section("stats"):
        stats = self.archive.stats()

    with self._profiler.section("journal"):
        self.evolution_journal.finalize_generation(...)

    with self._profiler.section("validation"):
        if best_genome:
            val_stats = self.run_validation(best_genome, gen)

    with self._profiler.section("checkpoint"):
        if gen % ckpt_freq == 0:
            self.save_checkpoint(...)

    # Live output
    if self._profiler.enabled:
        self.logger.info(f"Gen {gen}: {self._profiler.end_generation()}")

# After loop ends:
if self._profiler.enabled:
    self.logger.info(self._profiler.summary())
```

## Output Examples

### Live Per-Generation Output

```
Gen 0: [1523ms | GPU:2840MB] breeding=45ms | evaluation=1432ms(+14.2MB) | archive=23ms(-13.8MB) | stats=12ms | validation=0ms
Gen 1: [1247ms | GPU:2841MB] breeding=42ms | evaluation=1180ms(+12.3MB) | archive=12ms(-11.8MB) | stats=8ms | validation=0ms
Gen 2: [1298ms | GPU:2842MB] breeding=38ms | evaluation=1225ms(+11.9MB) | archive=15ms(-11.5MB) | stats=9ms | validation=0ms
...
Gen 10: [1892ms | GPU:2844MB] breeding=41ms | evaluation=1201ms | archive=14ms | stats=8ms | validation=620ms(+1.2MB)
```

### Final Summary Output

```
=== Evolution Profile Summary (100 generations) ===
Phase            Time (s)     %       Avg/gen     Mem Δ (MB)
──────────────────────────────────────────────────────────────────────
evaluation         118.2s    94.7%      1182ms    +12.1 avg
breeding             4.1s     3.3%        41ms    +0.1 avg
archive              1.2s     1.0%        12ms    -11.8 avg
validation           0.6s     0.5%         6ms    +0.2 avg
stats                0.4s     0.3%         4ms    +0.0 avg
journal              0.2s     0.2%         2ms    +0.0 avg
checkpoint           0.1s     0.1%         1ms    +0.0 avg
──────────────────────────────────────────────────────────────────────
TOTAL              124.8s                1248ms

Memory trend: Gen 0: 2840MB → Gen 99: 2844MB (+4MB) ✓ No leak detected
```

## Implementation Tasks

1. **Create `profiler.py`** - New file with `GenerationProfiler` class
2. **Integrate into `map_elites.py`** - Add profiler initialization and section wrappers
3. **Test with `EVOLUTION_PROFILE=1`** - Run a short evolution and verify output
4. **Optional: Add to base orchestrator** - If other orchestrators need profiling

## Future Enhancements (Not in Scope)

- Drill-down into evaluation phase (per-query breakdown via `StepProfiler` integration)
- Tensor allocation counting (`torch.cuda.memory_stats()`)
- Export to JSON/CSV for plotting
- Peak memory tracking per section (`torch.cuda.max_memory_allocated()`)
- Automatic bottleneck detection and recommendations

## Estimated Changes

| File | Changes |
|------|---------|
| `swarm_rag/evolution/execution/profiler.py` | New file (~120 lines) |
| `swarm_rag/evolution/orchestrators/map_elites.py` | ~30 lines modified |
