"""Persistent, content-addressed embeddings. Modified 2026-10-30, AGPL-3.0.

Only vectors, hashes and timestamps are stored; abstracts and API keys are not.
"""
from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time

from loguru import logger


PROJECT_ROOT = Path(os.environ.get('GITHUB_WORKSPACE') or Path(__file__).resolve().parents[3])
DEFAULT_PATH = PROJECT_ROOT / '.cache' / 'paper-embeddings' / 'vectors.sqlite3'


def _plain(value):
    if isinstance(value, Mapping) or hasattr(value, 'items'):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return [_plain(v) for v in value]


def _valid_vector(value):
    if not isinstance(value, list) or not value:
        return False
    try:
        if not all(type(x) in (int, float) and math.isfinite(x) for x in value):
            return False
    except (TypeError, ValueError, OverflowError):
        return False
    return any(x != 0 for x in value)


class VectorStore:
    def __init__(self, path):
        self.connection = None
        self.dirty = False
        path = Path(path)
        self.path = path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(2):
                try:
                    self.connection = sqlite3.connect(path, timeout=30)
                    self.connection.execute('PRAGMA journal_mode=DELETE')
                    self.connection.execute(
                        'CREATE TABLE IF NOT EXISTS embeddings ('
                        'namespace TEXT NOT NULL, digest TEXT NOT NULL, '
                        'vector TEXT NOT NULL, updated REAL NOT NULL, '
                        'PRIMARY KEY (namespace, digest))'
                    )
                    columns = {row[1] for row in self.connection.execute('PRAGMA table_info(embeddings)')}
                    if columns != {'namespace', 'digest', 'vector', 'updated'}:
                        raise sqlite3.DatabaseError('Incompatible embedding cache schema.')
                    self.connection.commit()
                    break
                except sqlite3.DatabaseError as exc:
                    self.close()
                    if attempt or not path.exists():
                        raise
                    # Preserve an unreadable cache outside the saved file path.
                    path.rename(path.with_name(path.name + f'.broken-{time.time_ns()}'))
                    logger.warning(f'Embedding cache unreadable; rebuilding it: {exc}')
        except (OSError, sqlite3.Error) as exc:
            self.close()
            logger.warning(f'Embedding cache unavailable; computing without persistence: {exc}')

    def read(self, namespace, digest):
        if self.connection is None:
            return None
        try:
            row = self.connection.execute(
                'SELECT vector FROM embeddings WHERE namespace=? AND digest=?',
                (namespace, digest),
            ).fetchone()
            if row:
                value = json.loads(row[0])
                if _valid_vector(value):
                    return value
        except (ValueError, TypeError, sqlite3.Error):
            pass
        return None

    def write(self, namespace, vectors):
        if self.connection is None or not vectors:
            return
        try:
            with self.connection:
                self.connection.executemany(
                    'INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?)',
                    [(namespace, digest, json.dumps(vector, allow_nan=False), time.time())
                     for digest, vector in vectors.items()],
                )
            self.mark_updated()
        except (sqlite3.Error, OSError) as exc:
            logger.warning(f'Could not save embedding batch: {exc}')

    def clear(self, namespace):
        if self.connection:
            try:
                with self.connection:
                    self.connection.execute('DELETE FROM embeddings WHERE namespace=?', (namespace,))
                self.mark_updated()
            except sqlite3.Error as exc:
                logger.warning(f'Could not reset embedding cache; computing without persistence: {exc}')
                self.close()

    def mark_updated(self):
        self.dirty = True
        try:
            (self.path.parent / '.updated').touch()
        except OSError as exc:
            logger.warning(f'Could not mark embedding cache update: {exc}')

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None


def cached_embeddings(config, backend, settings, texts, compute, batch_size=32):
    """Return vectors in input order; compute and commit only cache misses.

    `compute` receives a list of texts and returns an equally long list of vectors.
    Model construction can be lazy inside it, so a complete cache hit needs no model.
    """
    if not texts:
        return []
    options = config.get('embedding_cache') or {}
    enabled = options.get('enabled', True)
    path = Path(options.get('path') or DEFAULT_PATH)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    namespace = hashlib.sha256(json.dumps({
        'format': 1, 'backend': backend, 'settings': _plain(settings),
        'version': _plain(options.get('version', 1)),
    }, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    # Abstract changes alter the digest; duplicate abstracts share one vector.
    digests = [hashlib.sha256(text.encode('utf-8')).hexdigest() for text in texts]
    unique = dict(zip(digests, texts))
    store = VectorStore(path) if enabled else None
    try:
        vectors = {}
        if store:
            for digest in unique:
                vector = store.read(namespace, digest)
                if vector is not None:
                    vectors[digest] = vector
        if len({len(v) for v in vectors.values()}) > 1:
            logger.warning('Embedding cache dimensions inconsistent; rebuilding namespace.')
            store.clear(namespace)
            vectors = {}
        missing = [(d, t) for d, t in unique.items() if d not in vectors]
        logger.info(
            f'Embedding cache [{backend}]: {len(vectors)} hits, '
            f'{len(missing)} misses, {len(unique)} unique abstracts'
        )
        size = max(1, int(batch_size))

        def fill(pending):
            for start in range(0, len(pending), size):
                batch = pending[start:start + size]
                logger.info(f'Encoding uncached abstracts: {start + 1}-{start + len(batch)}/{len(pending)}')
                result = list(compute([text for _, text in batch]))
                if len(result) != len(batch):
                    raise ValueError('Embedding backend returned the wrong number of vectors.')
                fresh = {digest: [float(x) for x in vector]
                         for (digest, _), vector in zip(batch, result)}
                if not all(_valid_vector(v) for v in fresh.values()):
                    raise ValueError('Embedding backend returned an empty, zero or non-finite vector.')
                if len({len(v) for v in fresh.values()}) != 1:
                    raise ValueError('Embedding backend returned inconsistent dimensions.')
                if vectors and len(next(iter(vectors.values()))) != len(next(iter(fresh.values()))):
                    raise DimensionChanged('Embedding dimensions changed.')
                vectors.update(fresh)
                if store:
                    # A completed batch remains reusable even if a later request fails.
                    store.write(namespace, fresh)

        try:
            fill(missing)
        except DimensionChanged:
            logger.warning('Embedding dimensions changed; rebuilding all requested vectors once.')
            if store:
                store.clear(namespace)
            vectors.clear()
            fill(list(unique.items()))
        return [vectors[digest] for digest in digests]
    finally:
        if store:
            store.close()


class DimensionChanged(ValueError):
    pass
