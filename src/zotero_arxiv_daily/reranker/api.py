# Modified 2026-10-30: persistent API embeddings with content-based invalidation. AGPL-3.0.
from .base import BaseReranker, register_reranker
from .embedding_cache import cached_embeddings
from openai import OpenAI
import numpy as np


@register_reranker('api')
class ApiReranker(BaseReranker):
    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        api = self.config.reranker.api
        settings = {
            'base_url': str(api.base_url or 'https://api.openai.com/v1').rstrip('/'),
            'model': api.model,
        }
        client = None

        def compute(texts):
            nonlocal client
            if client is None:
                client = OpenAI(api_key=api.key, base_url=api.base_url)
            response = client.embeddings.create(input=texts, model=api.model)
            # Preserve the request ordering even if a provider returns a shuffled list.
            data = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in data] != list(range(len(texts))):
                raise ValueError('Embedding API returned invalid vector indices.')
            return [item.embedding for item in data]

        features = np.asarray(cached_embeddings(
            self.config, 'api', settings, s1 + s2, compute,
            batch_size=api.get('batch_size') or 64,
        ))
        left, right = features[:len(s1)], features[len(s1):]
        left = left / np.linalg.norm(left, axis=1, keepdims=True)
        right = right / np.linalg.norm(right, axis=1, keepdims=True)
        return np.dot(left, right.T)
