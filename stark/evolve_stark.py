"""
Evolutionary optimization for SwarmRAG on STaRK datasets.

Uses MAP-Elites with optional LLM-guided mutations to evolve
specialized retrieval strategies.
"""
import argparse
import os
import json
import random
import sys

# Import core components
from swarm_rag.core import SwarmRetriever
from swarm_rag.evolution.engine import EvolutionEngine
from swarm_rag.evolution.execution.fitness import FitnessCalculator
from swarm_rag.eval import Evaluator

# Import config types
from swarm_rag.evolution.types.config import (
    EvolutionConfig,
    MapElitesConfig,
    GeneticConfig,
    LLMConfig,
    ResourceConfig,
    CheckpointConfig,
)

# Import STaRK integrations
from swarm_rag.integrations.stark import (
    StarkInMemoryVectorStore,
    StarkPreComputedEmbeddingHandler,
    StarkSKBAdapter,
    create_stark_vector_store,  # GPU-aware factory
)
from load_stark import (
    load_and_download_embeddings,
    load_and_download_skb,
    load_and_download_qa,
    precompute_stark_adjacency,
)

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_preset(preset_name: str) -> dict:
    """
    Load a named preset from presets.yaml.

    Args:
        preset_name: Name of the preset (e.g., "toy", "fast", "full")

    Returns:
        Dictionary of preset configuration values
    """
    try:
        import yaml
    except ImportError:
        print("Error: PyYAML is required for preset loading. Install with: pip install pyyaml")
        sys.exit(1)

    preset_path = os.path.join(BASE_DIR, "presets.yaml")

    if not os.path.exists(preset_path):
        raise FileNotFoundError(f"Presets file not found: {preset_path}")

    with open(preset_path, "r") as f:
        presets = yaml.safe_load(f)

    if preset_name not in presets:
        available = list(presets.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")

    print(f"Loading preset: {preset_name}")
    return presets[preset_name]


def prepare_stark_data(dataset_name: str, split: str, sample_size: int = None):
    """
    Loads STaRK data and converts it into the format expected by EvolutionEngine.

    Returns: (queries_text, query_ids, ground_truth_ids)
    """
    print(f"Loading {dataset_name} ({split})...")
    raw_data = load_and_download_qa(dataset_name)
    subset = list(raw_data.get_subset(split))
    random.seed(42)

    if sample_size is not None and sample_size < len(subset):
        data = random.sample(subset, sample_size)
    else:
        random.shuffle(subset)
        data = subset

    queries = [item[0] for item in data]
    query_ids = [item[1] for item in data]
    answer_ids = [item[2] for item in data]

    return queries, query_ids, answer_ids


def run_evolution(
    dataset_name: str = "prime",
    n_gens: int = 100,
    pop_size: int = 30,
    initial_fill: int = 100,
    train_sample_size: int = 200,
    val_sample_size: int = 100,
    start_from_scratch: bool = False,
    concurrent_evals: int = 4,
    max_workers: int = 4,
    llm_enabled: bool = False,
    llm_provider: str = "cerebras",
    llm_model: str = "zai-glm-4.7",
    env_path: str = ".env",
    mutation_strategy: str = "guided_mutation",
    use_gpu: str = "auto",  # GPU mode: "auto", "always", "never"
):
    """
    Run MAP-Elites evolutionary optimization.

    Args:
        dataset_name: STARK dataset to use (prime, amazon, mag)
        n_gens: Number of generations
        pop_size: Batch size for offspring generation
        initial_fill: Initial population size to seed archive
        train_sample_size: Number of training samples
        val_sample_size: Number of validation samples
        start_from_scratch: Clear previous checkpoints/logs
        concurrent_evals: Number of concurrent genome evaluations
        max_workers: Workers per retrieval
        llm_enabled: Enable LLM-guided mutations
        llm_provider: LLM provider (cerebras, openai, groq, etc.)
        llm_model: Model ID for the provider
        env_path: Path to .env file with API keys
        mutation_strategy: Mutation strategy name
        use_gpu: GPU mode - "auto" (detect), "always" (require), "never" (CPU only)
    """
    print(f"\n{'='*60}")
    print("MAP-ELITES EVOLUTION")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Generations: {n_gens}")
    print(f"Batch Size: {pop_size}")
    print(f"Initial Fill: {initial_fill}")
    print(f"LLM Enabled: {llm_enabled}")
    print(f"GPU Mode: {use_gpu}")
    if use_gpu != "never":
        try:
            from swarm_rag.utils.device import get_device
            print(f"GPU Device: {get_device()}")
        except ImportError:
            print("GPU Device: (utils not available)")
    print(f"{'='*60}\n")

    # Load Data
    skb = load_and_download_skb(dataset_name)
    adj_dict = precompute_stark_adjacency(skb, dataset_name)
    query_embs, doc_embs = load_and_download_embeddings(dataset_name)

    # Initialize Core Components
    vector_store = create_stark_vector_store(doc_embs, use_gpu=use_gpu)
    graph_cache_path = os.path.join(BASE_DIR, "adjacency_cache", f"graph_{dataset_name}.npz")
    graph_store = StarkSKBAdapter(
        skb, dataset_name, adjacency_dict=adj_dict, cache_path=graph_cache_path
    )
    embedding_provider = StarkPreComputedEmbeddingHandler(query_embs)

    # Create Retriever (enable GPU if not explicitly disabled)
    retriever = SwarmRetriever(
        vector_store=vector_store,
        graph_store=graph_store,
        embedding_provider=embedding_provider,
        cache_neighbors=False,
        cache_vectors=True,
        use_gpu=(use_gpu != "never"),
    )

    # Prepare Data Subsets
    train_q, train_q_ids, train_gt = prepare_stark_data(
        dataset_name, "train", sample_size=train_sample_size
    )
    val_q, val_q_ids, val_gt = prepare_stark_data(
        dataset_name, "val", sample_size=val_sample_size
    )

    print(f"Evolution Corpus: {len(train_q)} training queries, {len(val_q)} validation queries.")

    # Define Fitness Goals
    fitness_calc = FitnessCalculator(
        weights={
            "Hit@1": 0.25,
            "Hit@5": 0.25,
            "MRR": 0.25,
            "Recall@20": 0.25,
            "complexity": -0.0001,  # Slight penalty for bloat
        },
    )

    evaluator = Evaluator(k_values=[1, 5, 10, 20])

    # Setup directories
    log_dir = os.path.join(BASE_DIR, "logs")
    ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
    best_params_dir = os.path.join(BASE_DIR, "best_params")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(best_params_dir, exist_ok=True)

    # Build configuration using new dataclass structure
    config = EvolutionConfig(
        n_generations=n_gens,
        fitness_strategy="lexicographic",
        resources=ResourceConfig(
            concurrent_evaluations=concurrent_evals,
            max_workers_per_retrieval=max_workers,
        ),
        map_elites=MapElitesConfig(
            dimensions=["aggressiveness", "complexity"],
            bins=[15, 12],
            ranges=[(10.0, 150.0), (5.0, 60.0)],
            initial_fill=initial_fill,
            batch_size=pop_size,
        ),
        genetic=GeneticConfig(
            mutation_strategy=mutation_strategy,
            crossover_strategy="uniform_parameter_mix",
            base_mutation_rate=0.25,
            crossover_rate=0.6,
            expr_max_depth=5,
            n_agent_groups=3,
        ),
        llm=LLMConfig(
            enabled=llm_enabled,
            provider=llm_provider,
            model=llm_model,
            env_path=env_path,
        ),
        checkpoint=CheckpointConfig(
            log_path=os.path.join(log_dir, f"evo_{dataset_name}.jsonl"),
            plot_path=os.path.join(log_dir, f"plot_{dataset_name}.png"),
            checkpoint_path=os.path.join(ckpt_dir, f"ckpt_{dataset_name}.pkl"),
            validation_frequency=5,
            checkpoint_frequency=5,
            plot_title=f"{dataset_name.title()} MAP-Elites Evolution",
        ),
    )

    # Enable LLM mutation if requested
    if llm_enabled:
        config.genetic.mutation_strategy = "llm_mutation"

    # Cleanup if starting from scratch
    if start_from_scratch:
        print("\n[Scratch Mode] Clearing previous evolution data...")
        best_params_path = os.path.join(best_params_dir, f"best_params_{dataset_name}.json")
        files_to_remove = [
            config.checkpoint.log_path,
            config.checkpoint.plot_path,
            config.checkpoint.checkpoint_path,
            best_params_path,
        ]
        # Also clean up intermediate checkpoints
        for f in os.listdir(ckpt_dir):
            if f.startswith(f"ckpt_{dataset_name}_gen_"):
                files_to_remove.append(os.path.join(ckpt_dir, f))

        for fpath in files_to_remove:
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                    print(f"  Deleted: {fpath}")
                except Exception as e:
                    print(f"  Error deleting {fpath}: {e}")
        print("[Scratch Mode] Cleanup complete.\n")

    # Check for checkpoint resume
    ckpt_path = config.checkpoint.checkpoint_path
    if os.path.exists(ckpt_path) and not start_from_scratch:
        print(f"Resuming from checkpoint: {ckpt_path}")
        try:
            engine = EvolutionEngine.load_checkpoint(
                checkpoint_path=ckpt_path,
                retriever=retriever,
                fitness_calculator=fitness_calc,
                evaluator=evaluator,
                train_query_ids=train_q_ids,
                train_ground_truth=train_gt,
                val_query_ids=val_q_ids,
                val_ground_truth=val_gt,
                config=config,
            )
        except Exception as e:
            print(f"Failed to load checkpoint ({e}). Starting fresh.")
            engine = EvolutionEngine(
                retriever=retriever,
                fitness_calculator=fitness_calc,
                evaluator=evaluator,
                train_query_ids=train_q_ids,
                train_ground_truth=train_gt,
                val_query_ids=val_q_ids,
                val_ground_truth=val_gt,
                config=config,
            )
    else:
        engine = EvolutionEngine(
            retriever=retriever,
            fitness_calculator=fitness_calc,
            evaluator=evaluator,
            train_query_ids=train_q_ids,
            train_ground_truth=train_gt,
            val_query_ids=val_q_ids,
            val_ground_truth=val_gt,
            config=config,
        )

    print("Starting Evolution Loop...")
    try:
        best_genome = engine.optimize()
    finally:
        print("Cleaning up shared memory...")
        vector_store.close()

    print("\n" + "=" * 30)
    print("Evolution Complete. Best Genome:")
    if best_genome:
        best_genome.pretty_print()

        # Save the best params for easy copy-pasting
        best_params_path = os.path.join(best_params_dir, f"best_params_{dataset_name}.json")
        with open(best_params_path, "w") as f:
            json.dump(best_genome.to_dict(), f, indent=2)
        print(f"Saved best genome to {best_params_path}")
    else:
        print("No best genome found.")
    print("=" * 30)

    return best_genome


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MAP-Elites evolutionary optimization for SwarmRAG on STaRK datasets"
    )

    # Preset configuration
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Load named preset from presets.yaml (toy, fast, full, llm)",
    )

    # Dataset and sampling
    parser.add_argument(
        "--dataset",
        type=str,
        default="prime",
        help="Dataset name (prime, amazon, mag)",
    )
    parser.add_argument("--train_ss", type=int, default=200, help="Number of training samples")
    parser.add_argument("--val_ss", type=int, default=100, help="Number of validation samples")

    # Evolution parameters
    parser.add_argument("--gens", type=int, default=100, help="Number of generations")
    parser.add_argument("--pop", type=int, default=30, help="Batch size for MAP-Elites")
    parser.add_argument(
        "--init_fill", type=int, default=100, help="Initial random population for MAP-Elites"
    )

    # Execution parameters
    parser.add_argument(
        "--concurrent", type=int, default=4, help="Number of concurrent genomes to evaluate"
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of threads per retrieval")

    # LLM configuration
    parser.add_argument("--llm", action="store_true", help="Enable LLM-guided mutations")
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="cerebras",
        help="LLM provider (cerebras, openai, groq, anthropic, etc.)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="zai-glm-4.7",
        help="Model ID for the LLM provider",
    )
    parser.add_argument(
        "--env-path", type=str, default=".env", help="Path to .env file with API keys"
    )

    # Mutation strategy
    parser.add_argument(
        "--mutation",
        type=str,
        default="guided_mutation",
        help="Mutation strategy (guided_mutation, expression_tree_mutation, aggressive_mutation)",
    )

    # GPU
    parser.add_argument(
        "--gpu",
        type=str,
        default="auto",
        choices=["auto", "always", "never"],
        help="GPU acceleration mode: auto (detect), always (require), never (CPU only)"
    )

    # Misc
    parser.add_argument("--scratch", action="store_true", help="Clear previous checkpoints/logs")

    args = parser.parse_args()

    # Apply preset if specified (CLI args override preset values)
    if args.preset:
        preset = load_preset(args.preset)

        # Only apply preset values if CLI arg was not explicitly provided
        if args.dataset == "prime":
            args.dataset = preset.get("dataset", args.dataset)
        if args.gens == 100:
            args.gens = preset.get("gens", args.gens)
        if args.pop == 30:
            args.pop = preset.get("pop", args.pop)
        if args.init_fill == 100:
            args.init_fill = preset.get("init_fill", args.init_fill)
        if args.train_ss == 200:
            args.train_ss = preset.get("train_ss", args.train_ss)
        if args.val_ss == 100:
            args.val_ss = preset.get("val_ss", args.val_ss)
        if args.concurrent == 4:
            args.concurrent = preset.get("concurrent", args.concurrent)
        if args.workers == 4:
            args.workers = preset.get("workers", args.workers)
        if args.mutation == "guided_mutation":
            args.mutation = preset.get("mutation_strategy", args.mutation)
        if args.llm_provider == "cerebras":
            args.llm_provider = preset.get("llm_provider", args.llm_provider)
        if args.llm_model == "zai-glm-4.7":
            args.llm_model = preset.get("llm_model", args.llm_model)
        if args.env_path == ".env":
            args.env_path = preset.get("env_path", args.env_path)

        # LLM preset override
        if not args.llm:
            args.llm = preset.get("llm_enabled", False)

        # GPU preset override
        if args.gpu == "auto":
            args.gpu = preset.get("use_gpu", args.gpu)

    run_evolution(
        dataset_name=args.dataset,
        n_gens=args.gens,
        pop_size=args.pop,
        initial_fill=args.init_fill,
        train_sample_size=args.train_ss,
        val_sample_size=args.val_ss,
        start_from_scratch=args.scratch,
        concurrent_evals=args.concurrent,
        max_workers=args.workers,
        llm_enabled=args.llm,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        env_path=args.env_path,
        mutation_strategy=args.mutation,
        use_gpu=args.gpu,
    )
