import os
import time
import cohere
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from stark_qa import load_qa              
from stark_experiments.core.swarm_retriever import SwarmRetriever 
from stark_experiments.core.metrics import Evaluator

def run_benchmark():
    load_dotenv(override=True)

    # --- CONFIGURATION ---
    DATASET = "prime"   # 'amazon', 'prime', or 'mag'
    SPLIT = "test"      # 'test', 'val', or 'train'
    SAMPLE_SIZE = 50    # Number of queries to run (None = all)
    DATA_DIR = "./data"
    # ---------------------

    print(f"--- Benchmarking Swarm on STaRK-{DATASET.upper()} ---")

    evaluator = Evaluator()

    print(f"Loading {SPLIT} set for {DATASET}...")
    qa_dataset = load_qa(DATASET, split=SPLIT, root=DATA_DIR)

    try:
        queries = qa_dataset.get_all_queries()
    except AttributeError:
        df = qa_dataset.data
        queries = list(zip(df['id'], df['query']))
    
    if SAMPLE_SIZE:
        queries = queries[:SAMPLE_SIZE]
        print(f"Subsampled to first {SAMPLE_SIZE} queries.")
    
    print("Initializing Swarm Retriever (Loading Graph & Vector DB)...")
    co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

    def embed_wrapper(texts):
        return co.embed(
            texts=texts, 
            model="embed-v4.0", 
            input_type="search_query"
        ).embeddings
    
    retriever = SwarmRetriever(
        db_path=os.path.join(DATA_DIR, "lancedb_storage"),
        table_name=f"stark_{DATASET}",
        graph_path=os.path.join(DATA_DIR, f"{DATASET}_edges.parquet"),
        embedding_fn=embed_wrapper
    )
    
    all_metrics = []

    print(f"Starting evaluation of {len(queries)} queries...")
    for q_id, q_text in tqdm(queries):
        start_time = time.time()
