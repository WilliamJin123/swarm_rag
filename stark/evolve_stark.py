"""
Evolutionary optimization for SwarmRAG on STaRK datasets.

Uses MAP-Elites with optional LLM-guided mutations to evolve
specialized retrieval strategies.

Target Metrics:
- Hit@1: >50%
- Hit@5: >70%
- MRR: >80%
- Recall@20: >80%
"""
import argparse
import os
import random
import sys

# Import core components
from swarm_rag.core import SwarmRetriever
from swarm_rag.evolution.engine import EvolutionEngine
from swarm_rag.evolution.execution.fitness import (
    FitnessCalculator,
    FitnessConfig,
    FitnessMode,
    MetricConfig,
    create_fitness_calculator,
)
from swarm_rag.eval import Evaluator

# Import config types
from swarm_rag.evolution.types.config import (
    EvolutionConfig,
    MapElitesConfig,
    GeneticConfig,
    LLMConfig,
    ResourceConfig,
    StorageConfig,
)

# Import storage
from swarm_rag.evolution.storage import RunManager

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


# ============ CONFIGURATION (edit manually between runs) ============
CONFIG = {
    # MAP-Elites parameters
    "pop_size": 30,
    "initial_fill": 100,

    # Resource allocation (hardware-dependent)
    "concurrent_evals": 4,
    "max_workers": 4,

    # LLM settings (used when --llm is passed)
    "llm_provider": "cerebras",
    "llm_model": "zai-glm-4.7",
    "env_path": ".env",

    # Evolution strategies
    "mutation_strategy": "guided_mutation",
    "creation_strategy": "baseline_seeded_initialization",
    "fitness_mode": "hybrid",

    # GPU mode: "auto", "always", "never"
    "use_gpu": "auto",
}
# ====================================================================


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


def list_runs(dataset: str = None):
    """List all existing runs, optionally filtered by dataset."""
    runs_dir = os.path.join(BASE_DIR, "runs")
    runs = RunManager.list_runs(runs_dir, dataset)

    if not runs:
        print("No runs found.")
        return

    print(f"\n{'Dataset':<12} {'Run ID':<20} {'Path'}")
    print("-" * 70)
    for run in runs:
        print(f"{run['dataset']:<12} {run['run_id']:<20} {run['path']}")
    print()


def run_evolution(
    dataset_name: str = "prime",
    run_id: str = None,
    resume_run: str = None,
    n_gens: int = 100,
    pop_size: int = 30,
    initial_fill: int = 100,
    train_sample_size: int = 200,
    val_sample_size: int = 100,
    concurrent_evals: int = 4,
    max_workers: int = 4,
    llm_enabled: bool = False,
    llm_provider: str = "cerebras",
    llm_model: str = "zai-glm-4.7",
    env_path: str = ".env",
    mutation_strategy: str = "guided_mutation",
    creation_strategy: str = "baseline_seeded_initialization",
    use_gpu: str = "auto",
    fitness_mode: str = "hybrid",
):
    """
    Run MAP-Elites evolutionary optimization.

    Target Metrics:
    - Hit@1: >50%
    - Hit@5: >70%
    - MRR: >80%
    - Recall@20: >80%

    Args:
        dataset_name: STARK dataset to use (prime, amazon, mag)
        run_id: Custom run identifier (auto-generated if None)
        resume_run: Path to existing run directory to resume
        n_gens: Number of generations
        pop_size: Batch size for offspring generation
        initial_fill: Initial population size to seed archive
        train_sample_size: Number of training samples
        val_sample_size: Number of validation samples
        concurrent_evals: Number of concurrent genome evaluations
        max_workers: Workers per retrieval
        llm_enabled: Enable LLM-guided mutations
        llm_provider: LLM provider (cerebras, openai, groq, etc.)
        llm_model: Model ID for the provider
        env_path: Path to .env file with API keys
        mutation_strategy: Mutation strategy name
        creation_strategy: Strategy for initial population
        use_gpu: GPU mode - "auto" (detect), "always" (require), "never" (CPU only)
        fitness_mode: Fitness calculation mode (hybrid recommended)
    """
    print(f"\n{'='*60}")
    print("MAP-ELITES EVOLUTION")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Generations: {n_gens}")
    print(f"Batch Size: {pop_size}")
    print(f"Initial Fill: {initial_fill}")
    print(f"Fitness Mode: {fitness_mode}")
    print(f"Creation Strategy: {creation_strategy}")
    print(f"Mutation Strategy: {mutation_strategy}")
    print(f"LLM Enabled: {llm_enabled}")
    print(f"GPU Mode: {use_gpu}")
    if use_gpu != "never":
        try:
            from swarm_rag.utils.device import get_device
            print(f"GPU Device: {get_device()}")
        except ImportError:
            print("GPU Device: (utils not available)")
    print(f"\nTarget Metrics:")
    print(f"  Hit@1: >50%  |  Hit@5: >70%  |  MRR: >80%  |  Recall@20: >80%")
    print(f"{'='*60}\n")

    # Create storage config
    storage = StorageConfig(
        base_dir=os.path.join(BASE_DIR, "runs"),
        dataset=dataset_name,
        run_id=run_id,
        use_gpu=use_gpu,
        checkpoint_frequency=5,
        validation_frequency=5,
        keep_n_checkpoints=10,
        plot_title=f"{dataset_name.title()} MAP-Elites Evolution",
    )

    # Create RunManager
    run_manager = RunManager(storage)

    # Handle resume vs new run
    if resume_run:
        # Resuming from existing run - update storage config to point to that run
        if not os.path.exists(resume_run):
            print(f"Error: Resume path does not exist: {resume_run}")
            sys.exit(1)

        # Extract dataset and run_id from path
        parts = os.path.normpath(resume_run).split(os.sep)
        if len(parts) >= 2:
            storage.dataset = parts[-2]
            storage.run_id = parts[-1]
            storage._resolve_paths()
            run_manager = RunManager(storage)

        print(f"Resuming run: {resume_run}")
        print(f"  Dataset: {storage.dataset}")
        print(f"  Run ID: {storage.run_id}")
    else:
        print(f"New run: {storage.run_dir}")

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
    fitness_calc = create_fitness_calculator(
        mode=fitness_mode,
        weights={
            "Hit@1": 0.25,
            "Hit@5": 0.25,
            "MRR": 0.25,
            "Recall@20": 0.25,
        },
        thresholds={
            "Hit@1": 0.50,
            "Hit@5": 0.75,    # updated from 0.70
            "MRR": 0.70,      # updated from 0.80
            "Recall@20": 0.80,
        }
    )

    evaluator = Evaluator(k_values=[1, 5, 10, 20])

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
            creation_strategy=creation_strategy,
            mutation_strategy=mutation_strategy,
            crossover_strategy="uniform_parameter_mix",
            base_mutation_rate=0.20,
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
        storage=storage,
    )

    # Enable LLM mutation if requested
    if llm_enabled:
        config.genetic.mutation_strategy = "llm_mutation"

    # Check for checkpoint resume
    if resume_run:
        checkpoint_path = os.path.join(resume_run, "checkpoints", "latest.pkl")
        if os.path.exists(checkpoint_path):
            print(f"Resuming from checkpoint: {checkpoint_path}")
            try:
                engine = EvolutionEngine.load_checkpoint(
                    run_manager=run_manager,
                    checkpoint_path=checkpoint_path,
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
                    run_manager=run_manager,
                )
                engine.initialize_run()
        else:
            print(f"No checkpoint found at {checkpoint_path}. Starting fresh.")
            engine = EvolutionEngine(
                retriever=retriever,
                fitness_calculator=fitness_calc,
                evaluator=evaluator,
                train_query_ids=train_q_ids,
                train_ground_truth=train_gt,
                val_query_ids=val_q_ids,
                val_ground_truth=val_gt,
                config=config,
                run_manager=run_manager,
            )
            engine.initialize_run()
    else:
        # New run
        engine = EvolutionEngine(
            retriever=retriever,
            fitness_calculator=fitness_calc,
            evaluator=evaluator,
            train_query_ids=train_q_ids,
            train_ground_truth=train_gt,
            val_query_ids=val_q_ids,
            val_ground_truth=val_gt,
            config=config,
            run_manager=run_manager,
        )
        engine.initialize_run()

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
        print(f"Best genome saved to: {run_manager.config.best_genome_path}")
    else:
        print("No best genome found.")
    print("=" * 30)

    return best_genome


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MAP-Elites evolutionary optimization for SwarmRAG on STaRK datasets"
    )

    # Essential CLI flags
    parser.add_argument(
        "--dataset",
        type=str,
        default="prime",
        help="Dataset name (prime, amazon, mag)",
    )
    parser.add_argument("--gens", type=int, default=100, help="Number of generations")
    parser.add_argument("--train_ss", type=int, default=200, help="Number of training samples")
    parser.add_argument("--val_ss", type=int, default=100, help="Number of validation samples")
    parser.add_argument("--llm", action="store_true", help="Enable LLM-guided mutations")

    # New storage-related arguments
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Custom run identifier (auto-generated timestamp if not provided)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to run directory to resume (e.g., stark/runs/prime/20240123_143022)",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List all existing runs and exit",
    )

    args = parser.parse_args()

    # Handle --list-runs
    if args.list_runs:
        list_runs(args.dataset if args.dataset != "prime" else None)
        sys.exit(0)

    # Merge CLI args with CONFIG dict
    run_evolution(
        dataset_name=args.dataset,
        run_id=args.run_id,
        resume_run=args.resume,
        n_gens=args.gens,
        train_sample_size=args.train_ss,
        val_sample_size=args.val_ss,
        llm_enabled=args.llm,
        **CONFIG,
    )
