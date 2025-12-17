import os
import numpy as np
from dotenv import load_dotenv
from stark_qa import load_skb
from stark_experiments.core.ingestor import CohereEmbeddingWrapper, DataIngestor
from tqdm import tqdm

if __name__== "__main__":
    load_dotenv(override=True)
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")  
    DATASETS = ["amazon", "prime", "mag"]
    
    DATASET_NAME = "prime"
    DATA_DIR = "./data"
    LANCEDB_PATH = os.path.join(DATA_DIR, "lancedb_storage")
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print(f"--- Initializing STaRK-{DATASET_NAME.upper()} Ingestion ---")

    print("Loading SKB dataset (this downloads if missing)...")
    skb = load_skb(DATASET_NAME, download_processed=True, root=DATA_DIR)
    total_nodes = skb.num_nodes()
    print(f"Loaded {total_nodes} nodes.")

    embedder = CohereEmbeddingWrapper(api_key=COHERE_API_KEY)
    ingestor = DataIngestor(
        db_path=LANCEDB_PATH, 
        embedding_fn=embedder,
        batch_size=96 # Optimal for Cohere v3
    )
    
    
    def prime_text_builder(idx):
        return skb.get_doc_info(idx, add_rel=True)

    def prime_id_builder(idx):
        return str(skb.node_info[idx]['id'])
    
    def prime_meta_builder(idx):
        info = skb.node_info[idx]
        return {
            "node_type": info.get('type', 'unknown'),
            "source": DATASET_NAME,
            "name": info.get('name', '')
        }
    
    print("Starting Vector Ingestion (Auto-Resume Enabled)...")
    ingestor.ingest_vectors(
        table_name=f"stark_{DATASET_NAME}",
        dataset=range(total_nodes),
        text_builder=prime_text_builder,
        id_builder=prime_id_builder,
        metadata_builder=prime_meta_builder
    )

    print("Processing Graph Edges...")

    print("Building ID Map...")
    idx_to_real_id = {i: str(skb.node_info[i]['id']) for i in range(total_nodes)}

    edge_index = skb.edge_index.cpu().numpy()
    src_indices = edge_index[0]
    dst_indices = edge_index[1]

    print("Mapping edges to Real IDs (this is fast)...")

    mapped_src = [idx_to_real_id[i] for i in tqdm(src_indices, desc="Mapping Sources")]
    mapped_dst = [idx_to_real_id[i] for i in tqdm(dst_indices, desc="Mapping Destinations")]
    
    final_edges = list(zip(mapped_src, mapped_dst))
    
    graph_path = os.path.join(DATA_DIR, f"{DATASET_NAME}_edges.parquet")
    ingestor.save_edge_list(final_edges, graph_path)
    
    print("\n--- DONE ---")
    print(f"Vectors stored in: {LANCEDB_PATH}")
    print(f"Graph Structure stored in: {graph_path}")