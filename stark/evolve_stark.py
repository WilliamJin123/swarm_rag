import argparse
import random
import numpy as np
import torch
from typing import List

# Import your existing engine components
from swarm_rag.core import SwarmRetriever
from swarm_rag.evolution.engine import EvolutionEngine
from swarm_rag.evolution.extensions.niching import NichingExtension
from swarm_rag.evolution.extensions.immigration import RandomImmigrationExtension
from swarm_rag.evolution.execution.fitness import FitnessCalculator
from swarm_rag.eval import Evaluator

# Import STaRK integrations
from swarm_rag.evolution.types.config import DEFAULT_EVO_CONFIG, EvolutionConfigDict
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
        data = random.shuffle(subset)
    
    queries = [item[0] for item in data]
    query_ids = [item[1] for item in data]
    answer_ids = [item[2] for item in data]
    
    return queries, query_ids, answer_ids

def run_evolution(dataset_name="prime", n_gens=20, pop_size=30):
    # Load
    skb = load_and_download_skb(dataset_name)
    adj_dict = precompute_stark_adjacency(skb, dataset_name)
    query_embs, doc_embs = load_and_download_embeddings(dataset_name)
    
    # Initialize Core Components
    vector_store = StarkInMemoryVectorStore(doc_embs)
    graph_store = StarkSKBAdapter(skb, dataset_name, adjacency_dict=adj_dict)
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
    # We use a smaller subset for Training (Speed) and Validation (Reliability)
    train_q, train_q_ids, train_gt = prepare_stark_data(dataset_name, 'train', sample_size=100)
    val_q, val_q_ids, val_gt = prepare_stark_data(dataset_name, 'val', sample_size=50)
    
    print(f"Evolution Corpus: {len(train_q)} training queries, {len(val_q)} validation queries.")

    # Define Fitness Goals
    # Balance: High Recall is king, but we penalize excessive cost (visited nodes)
    fitness_calc = FitnessCalculator(
        weights={
            "Hit@1": 0.30,     
            "Hit@5": 0.15,     
            "MRR": 0.15,        
            "Recall@20": 0.15, 
            # Others (not benchmarkmaxxed) 
            "Hit@10": 0.07,    
            "Recall@5": 0.07, 
            "Hit@20": 0.05,     
            "Recall@10": 0.06,
            "complexity": -0.01,
            # "variance": -0.1, 
            # "latency_ms": -0.00005, 
            # Don't penalize these because we are using lexicographic fitness (these are accounted for in the secondary / tertiary scores) 
        },
    )
    
    evaluator = Evaluator(k_values=[1, 5, 10, 20])

    # Configure Engine
    evo_config: EvolutionConfigDict = DEFAULT_EVO_CONFIG.copy().update({
        "n_generations": n_gens,
        "population_size": pop_size,
        "concurrent_evaluations": 4, # Parallelize evaluations
        
        # Evolution constraints
        "expr_max_depth": 5,  # Don't let trees get too crazy
        "param_ranges": {
            "n_agents": (5, 30),
            "steps": (2, 6),     # STaRK usually needs 2-4 hops
            "decay": (0.1, 0.9),
            "initial_pool_size": (10, 50),
        },
        
        # Logging
        "log_path": f"./logs/evo_{dataset_name}.json",
        "plot_path": f"./logs/plot_{dataset_name}.png",
        "checkpoint_path": f"./checkpoints/ckpt_{dataset_name}.pkl",
        "validation_frequency": 5
    })

    # Initialize Extensions
    extensions = [
        # Prevent population from converging to a single local optimum
        NichingExtension(sigma_share=2.5, n_probes=8),
        # Inject fresh random genomes to maintain diversity
        RandomImmigrationExtension(rate=0.1)
    ]

    # Launch Engine
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
    best_genome = engine.optimize()
    
    print("\n" + "="*30)
    print("Evolution Complete. Best Genome:")
    print(best_genome)
    print("="*30)
    
    # Save specifically the best params for easy copy-pasting
    import json
    with open(f"best_params_{dataset_name}.json", "w") as f:
        # Helper to dump the genome cleanly
        json.dump(best_genome.to_dict(), f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="prime")
    parser.add_argument("--gens", type=int, default=20)
    parser.add_argument("--pop", type=int, default=30)
    args = parser.parse_args()
    
    run_evolution(args.dataset, args.gens, args.pop)