from typing import List
from ..interfaces.base import EmbeddingProvider
from ..utils import fail_on_missing_imports
import numpy as np

try:
    import cohere
    from tenacity import retry, stop_after_attempt, wait_exponential
except:
    fail_on_missing_imports(['cohere', 'tenacity'])

class CohereEmbeddingProvider(EmbeddingProvider):
    
    def __init__(self, api_key: str, model: str = "embed-v4.0", input_type: str = "search_document"):
        self.co = cohere.ClientV2(api_key=api_key)
        self.model = model
        self.input_type = input_type

    def embed_query(self, query: str) -> np.ndarray:
        return np.array(self.embed_query_batch([query])[0])

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60))
    def embed_query_batch(self, texts: list[str]) -> np.ndarray:
        response = self.co.embed(
            texts=texts,
            model=self.model,
            input_type=self.input_type
        )
        return np.array(response.embeddings.float_)
    
