#!/usr/bin/env python
"""
CLI entry point for running performance benchmark.

Validates the full evolution optimization stack achieves target performance:
- 500 generations in 3 hours
- Peak VRAM under 4GB

Usage:
    python -m swarm_rag.benchmark.run_benchmark
    python -m swarm_rag.benchmark.run_benchmark --population 100 --generations 100
    python -m swarm_rag.benchmark.run_benchmark --output results.json
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from .performance_benchmark import PerformanceBenchmark, BenchmarkConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run performance validation benchmark for SwarmRAG evolution stack",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--population",
        type=int,
        default=75,
        help="Population size for evolution (target range: 50-100)",
    )

    parser.add_argument(
        "--generations",
        type=int,
        default=500,
        help="Target number of generations",
    )

    parser.add_argument(
        "--time-limit",
        type=float,
        default=3.0,
        help="Time limit in hours for pass criteria",
    )

    parser.add_argument(
        "--memory-limit",
        type=float,
        default=4.0,
        help="Memory limit in GB for pass criteria",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=".planning/phases/06-performance-validation/benchmark-results.json",
        help="Path to write JSON benchmark results",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def setup_logging(verbose: bool = False):
    """Configure logging for benchmark run."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Reduce noise from some libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def main():
    """Main entry point for benchmark CLI."""
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    # Display banner
    print()
    print("=" * 80)
    print("SWARM-RAG PERFORMANCE VALIDATION BENCHMARK")
    print("=" * 80)
    print()
    print(f"Configuration:")
    print(f"  Population:   {args.population}")
    print(f"  Generations:  {args.generations}")
    print(f"  Time limit:   {args.time_limit}h")
    print(f"  Memory limit: {args.memory_limit}GB")
    print(f"  Output:       {args.output}")
    print()

    # Create benchmark configuration
    config = BenchmarkConfig(
        population_size=args.population,
        target_generations=args.generations,
        time_limit_hours=args.time_limit,
        memory_limit_gb=args.memory_limit,
    )

    # Create and run benchmark
    benchmark = PerformanceBenchmark(config)

    try:
        result = benchmark.run()
    except Exception as e:
        logger.error(f"Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)

    # Set output path in result
    result.report_path = args.output

    # Print console summary
    print()
    print(result.to_console_summary())

    # Write JSON report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    logger.info(f"JSON report written to {output_path}")

    # Exit with appropriate code
    if result.overall_pass:
        logger.info("Benchmark PASSED - system meets performance requirements")
        sys.exit(0)
    else:
        logger.warning("Benchmark FAILED - review results for bottlenecks")
        sys.exit(1)


if __name__ == "__main__":
    main()
