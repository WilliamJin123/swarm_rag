import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import itertools


def load_stark_prime(root_dir: str):
    """
    Downloads STaRK-Prime directly from Hugging Face raw files.
    Returns: nodes (List[Dict]), edges (List[Tuple])
    """
    print("Downloading STaRK-Prime raw files from HF...")

    repo = "snap-stanford/stark"

    try:
        node_path = hf_hub_download(repo_id=repo, filename="skb/prime/processed/node_info.csv", repo_type="dataset", local_dir=root_dir)
        rel_path = hf_hub_download(repo_id=repo, filename="skb/prime/processed/relation_info.csv", repo_type="dataset", local_dir=root_dir)
    except:
        # Fallback: simpler file structure sometimes used in their updates
        print("Standard path failed, trying fallback...")
        node_path = hf_hub_download(repo_id=repo, filename="skb/prime/node.csv", repo_type="dataset", local_dir=root_dir)
        rel_path = hf_hub_download(repo_id=repo, filename="skb/prime/relation.csv", repo_type="dataset", local_dir=root_dir)

    print("Loading Nodes...")
    df_nodes = pd.read_csv(node_path)

    nodes = []
    for _, row in tqdm(df_nodes.iterrows(), total=len(df_nodes), desc="Parsing Nodes"):
        text_content = f"{row.get('name', '')}. {row.get('definition', '')} {row.get('content', '')}"
        nodes.append({
            "id": str(row['id']), # Ensure string ID
            "text": text_content.strip(),
            "type": row.get('type', 'unknown'),
            "name": row.get('name', 'unknown')
        })
    
    print("Loading Edges...")
    df_rels = pd.read_csv(rel_path)

    edges = []
    for _, row in df_rels.iterrows():
        edges.append((str(row[0]), str(row[1]))) # Tuple (src, dst)

    return nodes, edges

def load_2wiki():
    """
    Constructs graph from 2WikiMultiHopQA (Entities + Hyperlinks context).
    """
    print("Loading 2WikiMultiHopQA from HF...")
    dataset = load_dataset("2wikimultihopqa", split="train")
    
    node_map = {}
    edges = set()
    
    for row in tqdm(dataset, desc="Constructing 2Wiki Graph"):
        context_titles = [c[0] for c in row['context']]
        
        # 1. Create Nodes (Articles)
        for title in context_titles:
            if title not in node_map:
                node_map[title] = {
                    "id": title,
                    "text": title, # Title is often enough, or join sentences if needed
                    "type": "article",
                    "name": title
                }

        for src, dst in itertools.permutations(context_titles, 2):
            edges.add((src, dst))
            
    return list(node_map.values()), list(edges)

def load_musique():
    """
    Constructs graph from MuSiQue (similar to 2Wiki).
    """
    print("Loading MuSiQue from HF...")
    dataset = load_dataset("musique", split="train")
    
    node_map = {}
    edges = set()
    
    for row in tqdm(dataset, desc="Constructing MuSiQue Graph"):
        paragraphs = row['paragraphs']
        
        doc_titles = [p['title'] for p in paragraphs]
        
        for p in paragraphs:
            t = p['title']
            if t not in node_map:
                node_map[t] = {
                    "id": t,
                    "text": p['paragraph_text'], # Full text is better here
                    "type": "paragraph",
                    "name": t
                }
        for src, dst in itertools.permutations(doc_titles, 2):
            edges.add((src, dst))

    return list(node_map.values()), list(edges)

def load_biokgbench(root_dir: str):
    """
    BioKGBench uses a custom graph file. 
    We will download the 'biokg' subset or raw file if available.
    """
    print("Loading BioKGBench...")

    dataset = load_dataset("AutoLab-Westlake/BioKGBench-Dataset", "kgcheck", split="test")
    
    node_map = {}
    edges = set()
    
    print("Constructing Graph from BioKGBench samples...")
    for row in tqdm(dataset):
        claim = row.get('claim', '')
        if claim:
            node_map[claim] = {"id": claim, "text": claim, "type": "claim", "name": claim}
    return list(node_map.values()), []

ADAPTERS = {
    "stark_prime": load_stark_prime,
    "2wiki": load_2wiki,
    "musique": load_musique,
    "biokgbench": load_biokgbench,
}