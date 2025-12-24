
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
        start_index = 0
        if table_name in self.db.table_names():
            table = self.db.open_table(table_name)
            existing_count = len(table)
            if existing_count > 0:
                print(f"Found existing table '{table_name}' with {existing_count} rows. Resuming...")
                start_index = existing_count
        else:
            table = None
        
        total_batches = (len(dataset) + self.batch_size - 1) // self.batch_size
        # E.g., if 96 items done (batch size 96), we start at index 96
        current_batch_start = (start_index // self.batch_size) * self.batch_size

        iterator = range(current_batch_start, len(dataset), self.batch_size)
        
        if verbose:
            # Update tqdm to reflect skipped batches
            initial_steps = current_batch_start // self.batch_size
            iterator = tqdm(iterator, total=total_batches, initial=initial_steps, desc=f"Ingesting {table_name}")
        
        for i in iterator:
            if i + self.batch_size <= start_index:
                continue

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
                    meta = batch_metadatas[k]
                    clean_meta = {k: (v if v is not None else "") for k, v in meta.items()}
                    
                    record = {
                        "id": str(batch_ids[k]), 
                        "vector": emb,
                        "text": batch_texts[k],
                        **clean_meta
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
    def save_edge_list(self, src_list, dst_list, path: str):
        """Saves graph structure to Parquet (Fast & Compressed)"""
        print(f"Saving graph structure to {path}...")
        df = pd.DataFrame({"source": src_list, "target": dst_list})
        
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
        return response.embeddings.float_