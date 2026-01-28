"""
Evolutionary optimization for SwarmRAG on STaRK datasets.

Uses MAP-Elites with optional LLM-guided mutations to evolve
specialized retrieval strategies.

Target Metrics:
- Hit@1: >50%
- Hit@5: >75%
- MRR: >70%
- Recall@20: >80%
"""
import argparse
import os
import random
import sys

from dotenv import load_dotenv
load_dotenv(override=True)  

from swarm_rag.core import SwarmRetriever
from swarm_rag.evolution.engine import EvolutionEngine
from swarm_rag.evolution.execution.fitness import create_fitness_calculator
from swarm_rag.eval import Evaluator
from swarm_rag.evolution.types.config import (
    EvolutionConfig,
    MapElitesConfig,
    GeneticConfig,
    LLMConfig,
    ResourceConfig,
    StorageConfig,
    STARK_FEATURES,
)
from swarm_rag.evolution.storage import RunManager
from swarm_rag.integrations.stark import (
    StarkGraphAdapter,
    StarkPreComputedEmbeddingHandler,
    StarkVectorStore,
)
from load_stark import (
    load_and_download_embeddings,
    load_and_download_skb,
    load_and_download_qa,
    precompute_stark_adjacency,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def prepare_data(dataset_name: str, split: str, sample_size: int = None):
    """Load STaRK data subset."""
    raw_data = load_and_download_qa(dataset_name)
    subset = list(raw_data.get_subset(split))
    random.seed(42)

    if sample_size and sample_size < len(subset):
        data = random.sample(subset, sample_size)
    else:
        random.shuffle(subset)
        data = subset

    return [d[0] for d in data], [d[1] for d in data], [d[2] for d in data]


def list_runs(dataset: str = None):
    """List existing runs."""
    runs = RunManager.list_runs(os.path.join(BASE_DIR, "runs"), dataset)
    if not runs:
        print("No runs found.")
        return

    print(f"\n{'Dataset':<12} {'Run ID':<20} {'Path'}")
    print("-" * 70)
    for r in runs:
        print(f"{r['dataset']:<12} {r['run_id']:<20} {r['path']}")


def run_evolution(args):
    """Run MAP-Elites evolution."""
    print(f"\n{'='*50}")
    print(f"MAP-ELITES: {args.dataset.upper()} | Mode: {args.mode}")
    print(f"Generations: {args.gens} | Train: {args.train_ss} | Val: {args.val_ss}")
    print(f"{'='*50}\n")

    # Storage config
    storage = StorageConfig(
        base_dir=os.path.join(BASE_DIR, "runs"),
        dataset=args.dataset,
        run_id=args.run_id,
        device=args.device,
        checkpoint_frequency=25 if args.mode == "weighted_sum" else 20,
        validation_frequency=10,
    )
    run_manager = RunManager(storage)

    # Handle resume
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"Error: Resume path not found: {args.resume}")
            sys.exit(1)
        parts = os.path.normpath(args.resume).split(os.sep)
        if len(parts) >= 2:
            storage.dataset, storage.run_id = parts[-2], parts[-1]
            storage._resolve_paths()
            run_manager = RunManager(storage)
        print(f"Resuming: {args.resume}")
    else:
        print(f"New run: {storage.run_dir}")

    # Load data
    print("Loading STaRK data...")
    skb = load_and_download_skb(args.dataset)
    adj_dict = precompute_stark_adjacency(skb, args.dataset)
    query_embs, doc_embs = load_and_download_embeddings(args.dataset)

    # Resolve device once and pass to all components
    from swarm_rag.utils.device import resolve_device
    resolved_device = resolve_device(args.device)
    print(f"Using device: {resolved_device}")

    # Initialize components with resolved device
    vector_store = StarkVectorStore(doc_embs, device=resolved_device)
    graph_store = StarkGraphAdapter(
        skb, args.dataset,
        adjacency_dict=adj_dict,
        cache_path=os.path.join(BASE_DIR, "adjacency_cache", f"graph_{args.dataset}.npz"),
        device=resolved_device,
    )
    embedding_provider = StarkPreComputedEmbeddingHandler(query_embs)

    retriever = SwarmRetriever(
        vector_store=vector_store,
        graph_store=graph_store,
        embedding_provider=embedding_provider,
        cache_neighbors=False,
        cache_vectors=True,
        device=resolved_device,
    )

    # Prepare queries
    train_q, train_ids, train_gt = prepare_data(args.dataset, "train", args.train_ss)
    val_q, val_ids, val_gt = prepare_data(args.dataset, "val", args.val_ss)
    print(f"Queries: {len(train_q)} train, {len(val_q)} val")

    # Fitness calculator
    fitness_calc = create_fitness_calculator(
        mode="hybrid",
        weights={"Hit@1": 0.25, "Hit@5": 0.25, "MRR": 0.25, "Recall@20": 0.25},
        thresholds={"Hit@1": 0.50, "Hit@5": 0.75, "MRR": 0.70, "Recall@20": 0.80},
    )
    evaluator = Evaluator(k_values=[1, 5, 10, 20])

    # Mode-specific config
    if args.mode == "weighted_sum":
        initial_fill, batch_size = 100, 30
    else:
        initial_fill, batch_size = 80, 25

    config = EvolutionConfig(
        genome_mode=args.mode,
        heuristic_features=STARK_FEATURES,
        n_generations=args.gens,
        fitness_strategy="lexicographic",
        resources=ResourceConfig(
            concurrent_evaluations=4,
            max_workers_per_retrieval=4,
            enable_shared_precompute=True,
            enable_cross_genome_metric_batch=True,
            early_exit_threshold=0.25,
        ),
        map_elites=MapElitesConfig(
            dimensions=["aggressiveness", "complexity"],
            bins=[15, 12],
            ranges=[(10.0, 150.0), (5.0, 60.0)],
            initial_fill=initial_fill,
            batch_size=batch_size,
        ),
        genetic=GeneticConfig(
            creation_strategy="baseline_seeded_initialization",
            mutation_strategy="llm_mutation" if args.llm else "guided_mutation",
            crossover_strategy="uniform_parameter_mix",
            base_mutation_rate=0.20,
            crossover_rate=0.6,
            n_agent_groups=3,
        ),
        llm=LLMConfig(enabled=args.llm, provider="cerebras", model="zai-glm-4.7"),
        storage=storage,
    )

    # Create or resume engine
    checkpoint_path = os.path.join(args.resume, "checkpoints", "latest.pkl") if args.resume else None

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        try:
            engine = EvolutionEngine.load_checkpoint(
                run_manager=run_manager,
                checkpoint_path=checkpoint_path,
                retriever=retriever,
                fitness_calculator=fitness_calc,
                evaluator=evaluator,
                train_query_ids=train_ids,
                train_ground_truth=train_gt,
                val_query_ids=val_ids,
                val_ground_truth=val_gt,
                config=config,
            )
        except Exception as e:
            print(f"Checkpoint load failed ({e}), starting fresh")
            engine = None
    else:
        engine = None

    if engine is None:
        engine = EvolutionEngine(
            retriever=retriever,
            fitness_calculator=fitness_calc,
            evaluator=evaluator,
            train_query_ids=train_ids,
            train_ground_truth=train_gt,
            val_query_ids=val_ids,
            val_ground_truth=val_gt,
            config=config,
            run_manager=run_manager,
        )
        engine.initialize_run()

    # Run evolution
    print("Starting evolution...")
    try:
        best = engine.optimize()
    finally:
        vector_store.close()

    print(f"\n{'='*30}")
    print("Evolution Complete")
    if best:
        best.pretty_print()
        print(f"Saved to: {run_manager.config.best_genome_path}")
    print("=" * 30)

    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAP-Elites evolution for SwarmRAG on STaRK")
    parser.add_argument("--dataset", default="prime", help="Dataset: prime, amazon, mag")
    parser.add_argument("--mode", default="weighted_sum", choices=["weighted_sum", "expression_tree"],
                        help="Genome mode: weighted_sum (fast linear) or expression_tree (symbolic)")
    parser.add_argument("--gens", type=int, default=100, help="Number of generations")
    parser.add_argument("--train_ss", type=int, default=100, help="Training sample size")
    parser.add_argument("--val_ss", type=int, default=50, help="Validation sample size")
    parser.add_argument("--llm", action="store_true", help="Enable LLM-guided mutations")
    parser.add_argument("--run-id", help="Custom run ID (auto-generated if not provided)")
    parser.add_argument("--resume", help="Path to run directory to resume")
    parser.add_argument("--list-runs", action="store_true", help="List existing runs and exit")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Device: auto (detect best), cuda, mps, or cpu")

    args = parser.parse_args()

    if args.list_runs:
        list_runs(args.dataset if args.dataset != "prime" else None)
        sys.exit(0)

    run_evolution(args)
