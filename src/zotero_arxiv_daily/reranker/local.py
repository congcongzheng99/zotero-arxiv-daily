# Modified 2026-10-30: persistent vectors and lazy local model loading. AGPL-3.0.
from .base import BaseReranker, register_reranker
from .embedding_cache import cached_embeddings
from importlib.metadata import version, PackageNotFoundError
import numpy as np


def _version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return 'unknown'


@register_reranker('local')
class LocalReranker(BaseReranker):
    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        local = self.config.reranker.local
        encode_kwargs = dict(local.encode_kwargs or {})
        settings = {
            'model': local.model, 'revision': local.get('revision'),
            'encode_kwargs': encode_kwargs,
            'sentence_transformers': _version('sentence-transformers'),
            'transformers': _version('transformers'), 'torch': _version('torch'),
        }
        encoder = None

        def compute(texts):
            nonlocal encoder
            if encoder is None:
                from sentence_transformers import SentenceTransformer
                kwargs = {'trust_remote_code': True}
                if local.get('revision'):
                    kwargs['revision'] = local.revision
                encoder = SentenceTransformer(local.model, **kwargs)
            kwargs = dict(encode_kwargs)
            kwargs.setdefault('show_progress_bar', True)
            features = encoder.encode(texts, **kwargs)
            if hasattr(features, 'detach'):
                features = features.detach().cpu().numpy()
            return np.asarray(features).tolist()

        features = np.asarray(cached_embeddings(
            self.config, 'local', settings, s1 + s2, compute,
            batch_size=encode_kwargs.get('batch_size') or 32,
        ))
        left, right = features[:len(s1)], features[len(s1):]
        left = left / np.linalg.norm(left, axis=1, keepdims=True)
        right = right / np.linalg.norm(right, axis=1, keepdims=True)
        return np.dot(left, right.T)
