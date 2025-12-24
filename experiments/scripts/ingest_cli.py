import os
import pandas as pd
from datasets import load_dataset
import argparse
from tqdm import tqdm
from dotenv import load_dotenv

from ..core.ingestor import CohereEmbeddingWrapper, DataIngestor
from ..core.ingest_adapter import ADAPTERS

ENV_PATH = "../.env"

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Ingest Knowledge Graphs into LanceDB")
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=ADAPTERS.keys(), help="Which dataset to ingest")
    parser.add_argument("--env", type=str, default=ENV_PATH, help="Path to .env file")

    load_dotenv(dotenv_path= ENV_PATH, override=True)

    args = parser.parse_args()
    
    load_dotenv(dotenv_path=args.env, override=True)

    DATA_DIR = f"./data_{args.dataset}"
    LANCEDB_PATH = os.path.join(DATA_DIR, "lancedb_storage")
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"=== Starting Ingestion for {args.dataset.upper()} ===")

    if args.dataset in ["stark_prime", "biokgbench"]:
        nodes, edges = ADAPTERS[args.dataset](DATA_DIR)
    else:
        nodes, edges = ADAPTERS[args.dataset]()

    print(f"Stats: {len(nodes)} Nodes, {len(edges)} Edges")

    embedder = CohereEmbeddingWrapper(api_key=os.getenv("COHERE_API_KEY"))
    ingestor = DataIngestor(db_path=LANCEDB_PATH, embedding_fn=embedder)
    
    print("Ingesting Vectors...")
    ingestor.ingest_vectors(
        table_name=f"{args.dataset}_nodes",
        dataset=nodes,
        text_builder=lambda x: x['text'],
        id_builder=lambda x: x['id'],
        metadata_builder=lambda x: {"type": x['type'], "name": x['name']}
    )

    print("Saving Graph Edges...")
    src_list, dst_list = zip(*edges) if edges else ([], [])
    ingestor.save_edge_list(src_list, dst_list, os.path.join(DATA_DIR, "edges.parquet"))
    
    print("=== Ingestion Complete ===")