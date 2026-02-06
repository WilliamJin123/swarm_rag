"""
Generation-level profiler with GPU memory tracking for evolution loops.

Enable with EVOLUTION_PROFILE=1 environment variable.
"""
import json
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
        'current_gen', '_has_cuda', '_gen_start_mem', 'max_generations'
    )

    def __init__(self, enabled: bool = False, max_generations: int = 100):
        self.enabled = enabled
        self.max_generations = max_generations
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

        # Trim old generations to prevent unbounded growth
        if len(self.generation_timings) > self.max_generations:
            oldest = min(self.generation_timings.keys())
            del self.generation_timings[oldest]
            if oldest in self.generation_memory:
                del self.generation_memory[oldest]

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
            f"{'Phase':<16} {'Time (s)':<12} {'%':<8} {'Avg/gen':<12} {'Mem Delta (MB)':<16}",
            "-" * 70,
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

        lines.append("-" * 70)
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
                        f"Memory trend: Gen {first_gen}: {start_mb:.0f}MB -> "
                        f"Gen {last_gen}: {end_mb:.0f}MB ({drift:+.0f}MB) "
                        f"{'[OK] ' + leak_status if abs(drift) < 10 else '[!] ' + leak_status}"
                    )

        return "\n".join(lines)

    def save(self, path: str) -> None:
        """Save profiling data to JSON file."""
        if not self.enabled or not self.generation_timings:
            return

        data = {
            "generations": {},
            "summary": self._compute_summary_dict()
        }

        for gen in sorted(self.generation_timings.keys()):
            timings = self.generation_timings[gen]
            memory = self.generation_memory.get(gen, {})

            gen_data = {
                "total_ms": sum(timings.values()),
                "sections": {}
            }

            for section, ms in timings.items():
                section_data = {"time_ms": ms}
                if section in memory:
                    before, after = memory[section]
                    section_data["memory_before_mb"] = before / 1024 / 1024
                    section_data["memory_after_mb"] = after / 1024 / 1024
                    section_data["memory_delta_mb"] = (after - before) / 1024 / 1024
                gen_data["sections"][section] = section_data

            data["generations"][gen] = gen_data

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def _compute_summary_dict(self) -> dict:
        """Compute summary statistics as a dict for JSON export."""
        if not self.generation_timings:
            return {}

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

        # Build sections summary
        sections_summary = {}
        for name, total_section_ms in section_totals.items():
            pct = (total_section_ms / total_ms * 100) if total_ms > 0 else 0
            avg_ms = total_section_ms / n_gens
            sections_summary[name] = {
                "total_s": total_section_ms / 1000,
                "pct": pct,
                "avg_ms": avg_ms
            }

        result = {
            "total_generations": n_gens,
            "total_time_s": total_s,
            "sections": sections_summary,
        }

        # Memory trend
        gens_sorted = sorted(self.generation_timings.keys())
        if len(gens_sorted) >= 2:
            first_gen = gens_sorted[0]
            last_gen = gens_sorted[-1]

            first_mem = self.generation_memory.get(first_gen, {})
            last_mem = self.generation_memory.get(last_gen, {})

            if first_mem and last_mem:
                first_section = list(first_mem.keys())[0] if first_mem else None
                last_section = list(last_mem.keys())[-1] if last_mem else None

                if first_section and last_section:
                    start_mb = first_mem[first_section][0] / 1024 / 1024
                    end_mb = last_mem[last_section][1] / 1024 / 1024
                    drift = end_mb - start_mb

                    result["memory_trend"] = {
                        "start_mb": start_mb,
                        "end_mb": end_mb,
                        "drift_mb": drift,
                        "leak_detected": abs(drift) >= 10
                    }

        return result

    def clear(self):
        """Reset all profiling data."""
        self.generation_timings.clear()
        self.generation_memory.clear()
