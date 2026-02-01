from stark_qa import load_qa, load_skb
from stark_qa.skb import SKB
from stark_qa.retrieval import STaRKDataset
import torch
import gdown
import pickle
import os
from typing import Dict, List
from tqdm import tqdm

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_and_download_skb(dataset_name: str = 'prime', root: str = None) -> SKB:
    if root is None:
        root = os.path.join(BASE_DIR, 'skb')
    dataset_path = os.path.join(root, dataset_name)
    download = not os.path.isdir(dataset_path)
    return load_skb(dataset_name, root, download_processed=download)

def load_and_download_qa(dataset_name: str = 'prime', root: str = None, human_gen = False) -> STaRKDataset:
    if root is None:
        root = os.path.join(BASE_DIR, 'qa')
    return load_qa(name=dataset_name, root=root, human_generated_eval=human_gen)

def load_and_download_embeddings(dataset_name: str, root: str = None, emb_model: str = 'text-embedding-ada-002') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load embeddings as pre-stacked tensors for memory-efficient initialization.

    Returns:
        (query_tensor, doc_tensor, query_ids, doc_ids) - all as torch.Tensor

    The tensors are pre-allocated and filled in-place to avoid intermediate list
    allocation. Original mmap'd dicts are deleted immediately after filling.
    """
    import gc

    if root is None:
        root = os.path.join(BASE_DIR, 'embeddings')
    PREDEFINED_IDS = {
        'prime': {
            'query': '1MshwJttPZsHEM2cKA5T13SIrsLeBEdyU',
            'doc': '16EJvCMbgkVrQ0BuIBvLBp-BYPaye-Edy'
        },
        'amazon': {
            'query': '1QZLhOa_Uh6_Xf85My88XIfOLnmD-wcuq',
            'doc': '18NU7tw_Tcyp9YobxKubLISBncwLaAiJz'
        },
        'mag': {
            'query': '1HSfUrSKBa7mJbECFbnKPQgd6HSsI8spT',
            'doc': '1oVdScsDRuEpCFXtWQcTAx7ycvOggWF17'
        }
    }
    if dataset_name not in PREDEFINED_IDS:
        raise ValueError(f"Dataset {dataset_name} not found. Choose from ['prime', 'amazon', 'mag']")

    base_dir = os.path.join(root, dataset_name, emb_model)
    query_dir = os.path.join(base_dir, "query")
    doc_dir = os.path.join(base_dir, "doc")

    os.makedirs(query_dir, exist_ok=True)
    os.makedirs(doc_dir, exist_ok=True)

    query_path = os.path.join(query_dir, "query_emb_dict.pt")
    doc_path = os.path.join(doc_dir, "candidate_emb_dict.pt")

    if not os.path.exists(query_path):
        print(f"Downloading {dataset_name} query embeddings...")
        url = f'https://drive.google.com/uc?id={PREDEFINED_IDS[dataset_name]["query"]}'
        gdown.download(url, query_path, quiet=False)

    if not os.path.exists(doc_path):
        print(f"Downloading {dataset_name} document embeddings...")
        url = f'https://drive.google.com/uc?id={PREDEFINED_IDS[dataset_name]["doc"]}'
        gdown.download(url, doc_path, quiet=False)

    # Load query embeddings (smaller, process first)
    print(f"Loading query embeddings from {base_dir}...")
    query_embs_dict = torch.load(query_path, mmap=True, weights_only=False)

    # Pre-allocate query tensor (avoid intermediate list)
    n_queries = len(query_embs_dict)
    query_sorted_ids = sorted(query_embs_dict.keys())
    query_ids = torch.tensor(query_sorted_ids, dtype=torch.long)

    # Get embedding dimension from first embedding
    first_emb = query_embs_dict[query_sorted_ids[0]]
    if isinstance(first_emb, torch.Tensor):
        emb_dim = first_emb.squeeze().shape[0]
    else:
        emb_dim = len(first_emb)

    print(f"Stacking {n_queries} query embeddings (dim={emb_dim})...")
    query_tensor = torch.empty((n_queries, emb_dim), dtype=torch.float32)
    for i, qid in enumerate(query_sorted_ids):
        vec = query_embs_dict[qid]
        if isinstance(vec, torch.Tensor):
            vec = vec.detach().cpu().squeeze()
        else:
            vec = torch.as_tensor(vec).squeeze()
        query_tensor[i] = vec

    # Free query dict before loading doc dict
    del query_embs_dict
    gc.collect()

    # Load doc embeddings
    print(f"Loading document embeddings from {base_dir}...")
    doc_embs_dict = torch.load(doc_path, mmap=True, weights_only=False)

    # Pre-allocate doc tensor
    n_docs = len(doc_embs_dict)
    doc_sorted_ids = sorted(doc_embs_dict.keys())
    doc_ids = torch.tensor(doc_sorted_ids, dtype=torch.long)

    print(f"Stacking {n_docs} document embeddings (dim={emb_dim})...")
    doc_tensor = torch.empty((n_docs, emb_dim), dtype=torch.float32)
    for i, did in enumerate(doc_sorted_ids):
        vec = doc_embs_dict[did]
        if isinstance(vec, torch.Tensor):
            vec = vec.detach().cpu().squeeze()
        else:
            vec = torch.as_tensor(vec).squeeze()
        doc_tensor[i] = vec

    # Free doc dict
    del doc_embs_dict
    gc.collect()

    print(f"Loaded tensors: query {query_tensor.shape}, doc {doc_tensor.shape}")
    return query_tensor, doc_tensor, query_ids, doc_ids

def precompute_stark_adjacency(skb: SKB, dataset_name: str, cache_dir: str = None) -> Dict[int, List[int]]:
    """
    Pre-compute and cache all neighbor lists from STARK SKB.
    This converts slow sparse tensor operations into fast dictionary lookups.
    
    Args:
        skb: STARK SKB object
        dataset_name: Name of dataset (for cache file naming)
        cache_dir: Directory to store cached adjacency lists
        
    Returns:
        Dictionary mapping node_id -> list of neighbor ids
    """
    if cache_dir is None:
        cache_dir = os.path.join(BASE_DIR, "adjacency_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{dataset_name}_adjacency.pkl")
    
    # Try to load from cache
    if os.path.exists(cache_path):
        print(f"Loading pre-computed adjacency from {cache_path}")
        with open(cache_path, 'rb') as f:
            adjacency_dict = pickle.load(f)
        print(f"Loaded {len(adjacency_dict)} node adjacency lists")
        return adjacency_dict
    
    # Pre-compute all adjacency lists
    print(f"Pre-computing adjacency lists for {dataset_name}...")
    print("This will take a few minutes but only needs to be done once.")
    
    # Get all node IDs that have valid info
    all_node_ids = [node_id for node_id in skb.node_info.keys() if skb.node_info[node_id] != ""]
    
    adjacency_dict = {}
    
    # Process in batches with progress bar
    for node_id in tqdm(all_node_ids, desc="Computing adjacency"):
        # Call the slow method once per node
        neighbors = skb.get_neighbor_nodes(node_id)
        adjacency_dict[node_id] = neighbors
    
    # Save to cache
    print(f"Saving adjacency cache to {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump(adjacency_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Pre-computed {len(adjacency_dict)} adjacency lists")
    
    # Print some statistics
    degrees = [len(neighbors) for neighbors in adjacency_dict.values()]
    if degrees:
        print(f"Degree statistics:")
        print(f"  Min: {min(degrees)}")
        print(f"  Max: {max(degrees)}")
        print(f"  Mean: {sum(degrees) / len(degrees):.1f}")
        print(f"  Median: {sorted(degrees)[len(degrees)//2]}")
    
    return adjacency_dict


if __name__=="__main__":
    # for dataset in ['prime', 'amazon', 'mag']:
        # load_and_download_skb(dataset)

    # load_and_download_skb('mag')

    # for dataset in ['prime', 'amazon', 'mag']:
    #     load_and_download_qa(dataset, human_gen=False)
    #     load_and_download_qa(dataset, human_gen=True)
    
    # for dataset in ['prime', 'amazon', 'mag']:
    #     query_embs, doc_embs = load_and_download_embeddings(dataset)

    #     print(f"Loaded {len(doc_embs)} document embeddings.")

    #     first_node_id = list(doc_embs.keys())[0]
    #     print(f"Embedding shape for node {first_node_id}: {doc_embs[first_node_id].shape}")

    for dataset in ['prime', 'amazon', 'mag']:
        skb = load_and_download_skb(dataset)

        adjacency_dict = precompute_stark_adjacency(skb, dataset)

        test_node = list(adjacency_dict.keys())[0]
        neighbors = adjacency_dict[test_node]
        print(f"\nNode {test_node} has {len(neighbors)} neighbors")
        print(f"First 10 neighbors: {neighbors[:10]}")