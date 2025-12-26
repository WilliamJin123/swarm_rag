from stark_qa import load_qa, load_skb
from stark_qa.skb import SKB
from stark_qa.retrieval import STaRKDataset
import os
import faiss
import torch
import gdown

def load_and_download_skb(dataset_name: str = 'prime', root: str = './skb') -> SKB:
    dataset_path = os.path.join(root, dataset_name)
    download = not os.path.isdir(dataset_path)
    return load_skb(dataset_name, root, download_processed=download)

def load_and_download_qa(dataset_name: str = 'prime', root: str = './qa', human_gen = False) -> STaRKDataset:
    
    return load_qa(name=dataset_name, root=root, human_generated_eval=human_gen)

def load_and_download_embeddings(dataset_name: str, root: str = './embeddings', emb_model: str = 'text-embedding-ada-002') -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Returns query_embs, doc_embs"""
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

    print(f"Loading embeddings from {base_dir}...")
    query_embs = torch.load(query_path)
    doc_embs = torch.load(doc_path)
    
    return query_embs, doc_embs


if __name__=="__main__":
    # for dataset in ['prime', 'amazon', 'mag']:
        # load_and_download_skb(dataset)

    # load_and_download_skb('mag')

    # for dataset in ['prime', 'amazon', 'mag']:
    #     load_and_download_qa(dataset, human_gen=False)
    #     load_and_download_qa(dataset, human_gen=True)
    
    for dataset in ['prime', 'amazon', 'mag']:
        query_embs, doc_embs = load_and_download_embeddings(dataset)

        print(f"Loaded {len(doc_embs)} document embeddings.")

        first_node_id = list(doc_embs.keys())[0]
        print(f"Embedding shape for node {first_node_id}: {doc_embs[first_node_id].shape}")