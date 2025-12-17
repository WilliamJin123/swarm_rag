
import lancedb
import cohere
import os
from typing import List, Any, Callable, Dict, Optional
import pandas as pd
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

EmbeddingFunction = Callable[[List[str]], List[List[float]]]
TextBuilder = Callable[[Any], str]
IdBuilder = Callable[[Any], str]
MetadataBuilder = Callable[[Any], Dict[str, Any]]

class DataIngestor:
    def __init__(
        self, 
        db_path: str, # Path to local folder
        embedding_fn: EmbeddingFunction,
        batch_size: int = 96
    ):
        """
        Args:
            collection: The ChromaDB collection instance.
            embedding_fn: A function that takes a list of strings and returns a list of vectors.
            batch_size: How many items to process/embed at once.
        """
        self.db = lancedb.connect(db_path)
        self.embed_fn = embedding_fn
        self.batch_size = batch_size
    
    def ingest_vectors(
        self, 
        table_name: str,
        dataset: List[Any], 
        text_builder: TextBuilder,
        id_builder: IdBuilder,
        metadata_builder: Optional[MetadataBuilder] = None,
        verbose: bool = True
    ):
        """
        Generic ingestion loop. Handles batching, text extraction, embedding, and DB insertion.
        """
        total_batches = (len(dataset) + self.batch_size - 1) // self.batch_size
        iterator = range(0, len(dataset), self.batch_size)
        
        if verbose:
            iterator = tqdm(iterator, total=total_batches, desc=f"Ingesting {table_name}")

        table = None
        for i in iterator:
            batch_items = dataset[i : i + self.batch_size]
            
            try:
                batch_texts = [text_builder(item) for item in batch_items]
                batch_ids = [id_builder(item) for item in batch_items]
                
                batch_metadatas = []
                if metadata_builder:
                    batch_metadatas = [metadata_builder(item) for item in batch_items]
                else:
                    batch_metadatas = [{} for _ in batch_items]
                    
                valid_pairs = [(j, t) for j, t in enumerate(batch_texts) if t.strip()]
                if not valid_pairs:
                    continue
                
                valid_indices, texts_to_embed = zip(*valid_pairs)

                try:
                    embeddings = self.embed_fn(list(texts_to_embed))
                except Exception as e:
                    print(f"CRITICAL: Embedding failed permanently at batch index {i}. Stopping.")
                    raise e
                
                records = []
                for k, emb in zip(valid_indices, embeddings):
                    record = {
                        "id": str(batch_ids[k]), 
                        "vector": emb,
                        "text": batch_texts[k],
                        **batch_metadatas[k]
                    }
                    records.append(record)
                
                if table is None:
                    # On first successful batch, create (or overwrite) the table
                    table = self.db.create_table(table_name, data=records, mode="overwrite")
                else:
                    table.add(records)
            except Exception as e:
                print(f"Error processing batch starting at index {i}: {e}")


    #TO FIX   
    def save_edge_list(self, edges, path: str):
        """Saves graph structure to Parquet (Fast & Compressed)"""
        print(f"Saving graph structure to {path}...")
        df = pd.DataFrame(edges, columns=["source", "target"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path, index=False)
        print(f"Saved {len(df)} edges.")


class CohereEmbeddingWrapper:
    
    def __init__(self, api_key: str, model: str = "embed-v4.0", input_type: str = "search_document"):
        self.co = cohere.ClientV2(api_key=api_key)
        self.model = model
        self.input_type = input_type

    # Retry up to 5 times, waiting 4s, 8s, 16s... if an error occurs
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60))
    def __call__(self, texts: List[str]) -> List[List[float]]:
        response = self.co.embed(
            texts=texts,
            model=self.model,
            input_type=self.input_type
        )
        return response.embeddings