from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def embed(self, texts: List[str]) -> np.ndarray:
        # Returns float32 numpy array (num_texts, embedding_dim)
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings
