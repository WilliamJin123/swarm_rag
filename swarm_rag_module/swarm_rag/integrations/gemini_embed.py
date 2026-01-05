from typing import List, Union, Any
from ..interfaces.abstract_classes import EmbeddingProvider
from ..utils import fail_on_missing_imports
import numpy as np

try:
    import google.genai as genai
    from tenacity import retry, stop_after_attempt, wait_exponential
except:
    fail_on_missing_imports(['google-generativeai', 'tenacity'])

class GeminiEmbeddingProvider(EmbeddingProvider):

    def __init__(self, api_key: str, model: str = "models/text-embedding-004"):
        genai.configure(api_key=api_key)
        self.model = model

    def embed_query(self, query: Union[str, Any]) -> np.ndarray:
        if not isinstance(query, str):
            query = str(query)
        return np.array(self.embed_query_batch([query])[0])

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60))
    def embed_query_batch(self, queries: List[Union[str, Any]]) -> List[np.ndarray]:
        # Convert to strings if needed
        texts = [str(q) if not isinstance(q, str) else q for q in queries]

        result = genai.embed_content(
            model=self.model,
            content=texts,
            task_type="retrieval_query"
        )

        # The result.embeddings is a list of dicts with 'values'
        embeddings = [np.array(emb['values']) for emb in result['embeddings']]
        return embeddings