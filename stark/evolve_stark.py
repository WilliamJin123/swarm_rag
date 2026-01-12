import argparse
import os
import json

# Import your existing engine components
from swarm_rag.core import SwarmRetriever
from swarm_rag.evolution.engine import EvolutionEngine
from swarm_rag.evolution.extensions.niching import NichingExtension
from swarm_rag.evolution.extensions.immigration import RandomImmigrationExtension
from swarm_rag.evolution.execution.fitness import FitnessCalculator
from swarm_rag.eval import Evaluator

# Import STaRK integrations
from swarm_rag.evolution.types.config import DEFAULT_EVO_CONFIG
from swarm_rag.integrations.stark import (
    StarkInMemoryVectorStore, 
    StarkPreComputedEmbeddingHandler, 
    StarkSKBAdapter
)
from load_stark import (
    load_and_download_embeddings, 
    load_and_download_skb, 
    load_and_download_qa, 
    precompute_stark_adjacency
)

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def prepare_stark_data(dataset_name: str, split: str, sample_size: int = None):
    """
    Loads STaRK data and converts it into the format expected by EvolutionEngine.
    Returns: (queries_text, ground_truth_ids)
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
    dataset_name="prime", 
    n_gens=100, 
    pop_size=60, 
    initial_fill=120,
    train_sample_size=200,
    val_sample_size=100,
    start_from_scratch=False,
    concurrent_evals=4,
    max_workers=4,
    use_map_elites=True
):
    # Load Data
    skb = load_and_download_skb(dataset_name)
    adj_dict = precompute_stark_adjacency(skb, dataset_name)
    query_embs, doc_embs = load_and_download_embeddings(dataset_name)
    
    # Initialize Core Components
    vector_store = StarkInMemoryVectorStore(doc_embs)
    graph_cache_path = os.path.join(BASE_DIR, "adjacency_cache", f"graph_{dataset_name}.npz")
    graph_store = StarkSKBAdapter(skb, dataset_name, adjacency_dict=adj_dict, cache_path=graph_cache_path)
    embedding_provider = StarkPreComputedEmbeddingHandler(query_embs)
    
    # Create Retriever 
    retriever = SwarmRetriever(
        vector_store=vector_store,
        graph_store=graph_store,
        embedding_provider=embedding_provider,
        cache_neighbors=False, # Stark Extension Handles this
        cache_vectors=True
    )
    
    # Prepare Data Subsets
    train_q, train_q_ids, train_gt = prepare_stark_data(dataset_name, 'train', sample_size=train_sample_size)
    val_q, val_q_ids, val_gt = prepare_stark_data(dataset_name, 'val', sample_size=val_sample_size)
    
    print(f"Evolution Corpus: {len(train_q)} training queries, {len(val_q)} validation queries.")

    # Define Fitness Goals
    # Heavily weight Hit@1 and MRR for precision, Recall@20 for broad search capabilities.
    fitness_calc = FitnessCalculator(
        weights={
            "Hit@1": 0.25,
            "Hit@5": 0.25,
            "MRR": 0.25,        
            "Recall@20": 0.25, 
            "complexity": -0.0001, # Slight penalty for bloat
        },
    )
    
    evaluator = Evaluator(k_values=[1, 5, 10, 20])

    log_dir = os.path.join(BASE_DIR, "logs")
    ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
    best_params_dir = os.path.join(BASE_DIR, "best_params")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(best_params_dir, exist_ok=True)

    # Configure Engine
    evo_config = DEFAULT_EVO_CONFIG.copy()
    
    # Common Config
    evo_config.update({
        "n_generations": n_gens,
        "concurrent_evaluations": concurrent_evals,
        "max_workers_per_retrieval": max_workers,
        
        # Paths
        "log_path": os.path.join(log_dir, f"evo_{dataset_name}.jsonl"),
        "plot_path": os.path.join(log_dir, f"plot_{dataset_name}.png"),
        "checkpoint_path": os.path.join(ckpt_dir, f"ckpt_{dataset_name}.pkl"),
        "validation_frequency": 5,
        "checkpoint_frequency": 5,
        
        # Constraints & Search Space
        "expr_max_depth": 5,
        "swarmrag_param_ranges": {
            "n_agents": (5, 30),
            "steps": (2, 8),
            "decay": (0.3, 0.95),
            "initial_pool_size": (10, 30),
            "start_subset": (5, 15),
        },
    })
    
    if use_map_elites:
        print(">> Mode: MAP-Elites Evolution")
        evo_config.update({
            "map_elites_enabled": True,
            "population_size": pop_size, # This becomes the batch size for offspring generation
            "map_elites_initial_fill": initial_fill,
            # Dimensions: 
            # 1. Aggressiveness (n_agents * steps): Measures "effort". Range ~10 to 240.
            # 2. Complexity (Tree Size): Measures "sophistication". Range ~5 to 50.
            "map_elites_dims": ["aggressiveness", "complexity"],
            "map_elites_bins": [15, 12], 
            "map_elites_ranges": [(10, 150), (5, 60)],
            
            # Genetic Operators
            "crossover_strategy": "uniform_parameter_mix", # Simple mixing
            "mutation_strategy": "guided_mutation",        # Smart mutation
            "base_mutation_rate": 0.25,
        })
    else:
        print(">> Mode: Standard Evolution (GA)")
        evo_config.update({
            "map_elites_enabled": False,
            "population_size": pop_size,
            "creation_strategy": "seeded_initialization",
            "crossover_strategy": "root_mix_crossover",
            "mutation_strategy": "guided_mutation",
            "base_mutation_rate": 0.2,
            "elite_fraction": 0.1,
        })
    
    # CLEANUP IF START_FROM_SCRATCH
    if start_from_scratch:
        print("\n[Scratch Mode] Clearing previous evolution data...")
        best_params_path = os.path.join(best_params_dir, f"best_params_{dataset_name}.json")
        files_to_remove = [
            evo_config["log_path"],
            evo_config["plot_path"],
            evo_config["checkpoint_path"],
            best_params_path
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


    # Initialize Extensions (Standard GA only)
    extensions = []
    if not use_map_elites:
        extensions = [
            NichingExtension(sigma_share=2.5, n_probes=8),
            RandomImmigrationExtension(rate=0.1)
        ]

    # Launch Engine
    # Check for checkpoint resume
    if os.path.exists(evo_config["checkpoint_path"]) and not start_from_scratch:
        print(f"Resuming from checkpoint: {evo_config['checkpoint_path']}")
        try:
            engine = EvolutionEngine.load_checkpoint(
                checkpoint_path=evo_config["checkpoint_path"],
                retriever=retriever,
                fitness_calculator=fitness_calc,
                evaluator=evaluator,
                train_query_ids=train_q_ids,
                train_ground_truth=train_gt,
                val_query_ids=val_q_ids,
                val_ground_truth=val_gt,
                config=evo_config,
                extensions=extensions
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
                config=evo_config,
                extensions=extensions
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
            config=evo_config,
            extensions=extensions
        )

    print("Starting Evolution Loop...")
    try:
        best_genome = engine.optimize()
    finally:
        print("Cleaning up shared memory...")
        vector_store.close()
    
    print("\n" + "="*30)
    print("Evolution Complete. Best Genome:")
    if best_genome:
        best_genome.pretty_print()
        
        # Save specifically the best params for easy copy-pasting
        best_params_path = os.path.join(best_params_dir, f"best_params_{dataset_name}.json")
        with open(best_params_path, "w") as f:
            json.dump(best_genome.to_dict(), f, indent=2)
        print(f"Saved best genome to {best_params_path}")
    else:
        print("No best genome found.")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="prime")
    parser.add_argument("--gens", type=int, default=100)
    parser.add_argument("--pop", type=int, default=30, help="Batch size for MAP-Elites or Pop Size for GA")
    parser.add_argument("--init_fill", type=int, default=100, help="Initial random population for MAP-Elites")
    parser.add_argument("--train_ss", type=int, default=200, help="Number of training samples.")
    parser.add_argument("--val_ss", type=int, default=100, help="Number of validation samples.")
    parser.add_argument("--concurrent", type=int, default=4, help="Number of concurrent genomes to evaluate.")
    parser.add_argument("--workers", type=int, default=4, help="Number of threads per retrieval.")
    parser.add_argument("--scratch", action="store_true", help="Clear previous checkpoints/logs.")
    parser.add_argument("--standard-ga", action="store_true", help="Use Standard GA instead of MAP-Elites")
    
    args = parser.parse_args()
    
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
        use_map_elites=(not args.standard_ga)
    )